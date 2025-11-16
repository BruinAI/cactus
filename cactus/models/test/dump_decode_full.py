import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_NAME = "openai/whisper-medium"
MEL_PATH = "mel.npy"
TOK_PATH = "decoder_input_tokens.npy"

def main():
    print("Loading model...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    # ----- Load inputs -----
    mel = torch.from_numpy(np.load(MEL_PATH)).float()      # (1, 80, T_mel)
    toks_np = np.load(TOK_PATH).astype("int64")            # (T_dec,)
    decoder_input_ids = torch.tensor(toks_np, dtype=torch.long)[None, :]  # (1, T_dec)

    # ----- Encoder -----
    with torch.no_grad():
        enc_out = model.model.encoder(mel).last_hidden_state  # (1, T_enc, 1024)

    decoder = model.model.decoder

    # ----- Decoder embed + pos -----
    with torch.no_grad():
        hidden_states = decoder.embed_tokens(decoder_input_ids)

        # embed_scale exists in Whisper
        if getattr(decoder, "embed_scale", None) is not None:
            hidden_states = hidden_states * decoder.embed_scale

        # positional embeddings (returns [1, 1, T_dec, 1024])
        positions = decoder.embed_positions(decoder_input_ids)
        hidden_states = hidden_states + positions  # -> (1, 1, T, 1024)

        # 🔧 FIX SHAPE: collapse (1,1,T,1024) → (1,T,1024)
        bsz, tgt_len = decoder_input_ids.shape
        hidden_states = hidden_states.view(bsz, tgt_len, -1)

        # --- Save emb + pos ---
        emb_pos = hidden_states[0].cpu().numpy()   # (T_dec, 1024)
        np.save("decoder_emb_pos_ref.npy", emb_pos)
        print("Saved decoder_emb_pos_ref.npy", emb_pos.shape)
        print("emb_pos tok0:", emb_pos[0, :5])
        print("emb_pos tok1:", emb_pos[1, :5])
        print("emb_pos tok2:", emb_pos[2, :5])


        # ----- Decoder layers (manual forward) -----
        x = hidden_states
        block0_out = None

        for i, layer in enumerate(decoder.layers):

            # ---- Self-attention block ----
            residual = x
            x_ln1 = layer.self_attn_layer_norm(x)

            self_attn_out, _, _ = layer.self_attn(
                x_ln1,
                past_key_value=None,
                attention_mask=None,      # Whisper uses no mask here
                layer_head_mask=None,
                output_attentions=False,
            )
            x = residual + self_attn_out

            # ---- Cross-attention block ----
            residual = x
            x_ln2 = layer.encoder_attn_layer_norm(x)

            cross_out, _, _ = layer.encoder_attn(
                x_ln2,
                key_value_states=enc_out,
                past_key_value=None,
                attention_mask=None,
                layer_head_mask=None,
                output_attentions=False,
            )
            x = residual + cross_out

            # ---- Feed Forward block ----
            residual = x
            x_ln3 = layer.final_layer_norm(x)
            ffn = layer.fc2(layer.activation_fn(layer.fc1(x_ln3)))
            x = residual + ffn

            # Save layer 0 output
            if i == 0:
                block0_out = x[0].cpu().numpy()
                np.save("decoder_block0_out_ref.npy", block0_out)
                print("Saved decoder_block0_out_ref.npy", block0_out.shape)
                print("block0_out tok0:", block0_out[0, :5])
                print("block0_out tok1:", block0_out[1, :5])
                print("block0_out tok2:", block0_out[2, :5])


        # ----- Before final layer_norm -----
        dec_pre_norm = x[0].cpu().numpy()
        np.save("decoder_pre_norm_ref.npy", dec_pre_norm)
        print("Saved decoder_pre_norm_ref.npy", dec_pre_norm.shape)
        print("pre_norm tok0:", dec_pre_norm[0, :5])
        print("pre_norm tok1:", dec_pre_norm[1, :5])
        print("pre_norm tok2:", dec_pre_norm[2, :5])

        # ----- After final layer_norm -----
        dec_post_norm = decoder.layer_norm(x)[0].cpu().numpy()
        np.save("decoder_post_norm_ref.npy", dec_post_norm)
        print("Saved decoder_post_norm_ref.npy", dec_post_norm.shape)
        print("post_norm tok0:", dec_post_norm[0, :5])
        print("post_norm tok1:", dec_post_norm[1, :5])
        print("post_norm tok2:", dec_post_norm[2, :5])


if __name__ == "__main__":
    main()
