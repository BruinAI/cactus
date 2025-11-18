"""
Claude API client with batching support.

Supports two batching modes:
1. Claude Batch API (recommended for large batches, async processing)
2. Multithreaded parallel requests (for smaller batches, immediate results)
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class ClaudeAPIClient:
    """Client for calling Claude API with batching support."""

    def __init__(self, api_key: Optional[str] = None, max_workers: int = 10):
        """
        Initialize Claude API client.

        Args:
            api_key: Anthropic API key (if None, reads from ANTHROPIC_API_KEY env var)
            max_workers: Maximum number of threads for parallel requests
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in ANTHROPIC_API_KEY environment variable")

        self.client = Anthropic(api_key=self.api_key)
        self.max_workers = max_workers

    def call_single(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-3-5-haiku-20241022",
        max_tokens: int = 1024,
        temperature: float = 1.0,
        system: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make a single API call to Claude.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            system: Optional system prompt
            **kwargs: Additional parameters for the API

        Returns:
            Response dictionary with 'content', 'stop_reason', etc.
        """
        request_params = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
            **kwargs
        }

        if system:
            request_params["system"] = system

        response = self.client.messages.create(**request_params)

        return {
            "id": response.id,
            "content": response.content[0].text if response.content else "",
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        }

    def call_parallel(
        self,
        requests: List[Dict[str, Any]],
        callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        error_callback: Optional[Callable[[int, Exception], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Make multiple API calls in parallel using multithreading.

        Args:
            requests: List of request dicts, each containing parameters for call_single
            callback: Optional callback function(index, response) called on each completion
            error_callback: Optional callback function(index, exception) called on errors

        Returns:
            List of responses in the same order as requests
        """
        results = [None] * len(requests)

        def process_request(idx: int, request: Dict[str, Any]) -> tuple:
            """Process a single request and return (index, result)."""
            try:
                response = self.call_single(**request)
                if callback:
                    callback(idx, response)
                return (idx, response, None)
            except Exception as e:
                if error_callback:
                    error_callback(idx, e)
                return (idx, None, e)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_request, idx, req): idx
                for idx, req in enumerate(requests)
            }

            for future in as_completed(futures):
                idx, response, error = future.result()
                if error:
                    results[idx] = {"error": str(error)}
                else:
                    results[idx] = response

        return results

    def create_batch(
        self,
        requests: List[Dict[str, Any]],
        custom_ids: Optional[List[str]] = None
    ) -> str:
        """
        Create a batch using the Claude Batch API.

        Args:
            requests: List of request dicts, each containing parameters for call_single
            custom_ids: Optional list of custom IDs for each request

        Returns:
            Batch ID
        """
        if custom_ids and len(custom_ids) != len(requests):
            raise ValueError("custom_ids must have same length as requests")

        # Format requests for batch API
        batch_requests = []
        for idx, request in enumerate(requests):
            custom_id = custom_ids[idx] if custom_ids else f"request_{idx}"

            # Build the API parameters
            params = {
                "model": request.get("model", "claude-3-5-sonnet-20241022"),
                "max_tokens": request.get("max_tokens", 1024),
                "messages": request["messages"]
            }

            if "temperature" in request:
                params["temperature"] = request["temperature"]
            if "system" in request:
                params["system"] = request["system"]

            batch_requests.append({
                "custom_id": custom_id,
                "params": params
            })

        # Create batch
        batch = self.client.messages.batches.create(requests=batch_requests)

        return batch.id

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get the status of a batch.

        Args:
            batch_id: Batch ID

        Returns:
            Dictionary with batch status information
        """
        batch = self.client.messages.batches.retrieve(batch_id)

        return {
            "id": batch.id,
            "processing_status": batch.processing_status,
            "request_counts": {
                "processing": batch.request_counts.processing,
                "succeeded": batch.request_counts.succeeded,
                "errored": batch.request_counts.errored,
                "canceled": batch.request_counts.canceled,
                "expired": batch.request_counts.expired
            },
            "ended_at": batch.ended_at,
            "created_at": batch.created_at
        }

    def get_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """
        Get results from a completed batch.

        Args:
            batch_id: Batch ID

        Returns:
            List of results, one per request
        """
        results = []

        for result in self.client.messages.batches.results(batch_id):
            if result.result.type == "succeeded":
                message = result.result.message
                results.append({
                    "custom_id": result.custom_id,
                    "success": True,
                    "content": message.content[0].text if message.content else "",
                    "stop_reason": message.stop_reason,
                    "usage": {
                        "input_tokens": message.usage.input_tokens,
                        "output_tokens": message.usage.output_tokens
                    }
                })
            else:
                results.append({
                    "custom_id": result.custom_id,
                    "success": False,
                    "error": result.result.error
                })

        return results

    def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: int = 10,
        timeout: Optional[int] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Wait for a batch to complete and return results.

        Args:
            batch_id: Batch ID
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait (None for no timeout)
            callback: Optional callback function(status) called on each poll

        Returns:
            List of results from the completed batch
        """
        start_time = time.time()

        while True:
            status = self.get_batch_status(batch_id)

            if callback:
                callback(status)

            if status["processing_status"] == "ended":
                return self.get_batch_results(batch_id)

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Batch {batch_id} did not complete within {timeout} seconds")

            time.sleep(poll_interval)


def example_parallel():
    """Example usage of parallel requests."""
    client = ClaudeAPIClient()

    # Prepare multiple requests
    requests = [
        {
            "messages": [{"role": "user", "content": f"What is {i} + {i}?"}],
            "max_tokens": 100,
            "temperature": 0.0
        }
        for i in range(1, 6)
    ]

    # Progress callback
    def on_complete(idx, response):
        print(f"✓ Request {idx} completed: {response['content'][:50]}...")

    # Make parallel requests
    print("Making 5 parallel requests...")
    results = client.call_parallel(requests, callback=on_complete)

    print(f"\nCompleted {len(results)} requests")
    for idx, result in enumerate(results):
        if "error" in result:
            print(f"Request {idx}: ERROR - {result['error']}")
        else:
            print(f"Request {idx}: {result['content']}")


def example_batch():
    """Example usage of batch API."""
    client = ClaudeAPIClient()

    # Prepare batch requests
    requests = [
        {
            "messages": [{"role": "user", "content": f"What is {i} squared?"}],
            "max_tokens": 100,
            "temperature": 0.0
        }
        for i in range(1, 11)
    ]

    custom_ids = [f"math_problem_{i}" for i in range(1, 11)]

    # Create batch
    print("Creating batch...")
    batch_id = client.create_batch(requests, custom_ids)
    print(f"Batch created: {batch_id}")

    # Wait for completion
    def on_poll(status):
        print(f"Status: {status['processing_status']}, "
              f"Succeeded: {status['request_counts']['succeeded']}/{len(requests)}")

    print("Waiting for batch to complete...")
    results = client.wait_for_batch(batch_id, poll_interval=5, callback=on_poll)

    print(f"\nBatch completed with {len(results)} results")
    for result in results:
        if result["success"]:
            print(f"{result['custom_id']}: {result['content']}")
        else:
            print(f"{result['custom_id']}: ERROR - {result['error']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        example_batch()
    else:
        example_parallel()
