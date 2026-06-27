#%%

import sys; print(sys.executable)
import transformers; print(transformers.__version__)

from transformers import AutoModelForCausalLM, AutoTokenizer
print("core classes OK")

from transformers import BloomPreTrainedModel
print("BloomPreTrainedModel OK")
#%%

import contextlib
import gc
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import plotly.express as px
import pytest
import torch
from dotenv import load_dotenv
from IPython.display import display
from jaxtyping import Float, Int
from peft import LoraConfig
from torch import Tensor
from tqdm.notebook import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part34_activation_oracles"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

# Disable runtime errors from custom hooks
os.environ["TORCHDYNAMO_DISABLE"] = "1"
# Allow expandable memory segments on CUDA to avoid OOMs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import part34_activation_oracles.tests as tests
import part34_activation_oracles.utils as utils

MAIN = __name__ == "__main__"

dtype = torch.bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def print_with_wrap(s: str, width: int = 80):
    """Print text with line wrapping, preserving newlines."""
    out = []
    for line in s.splitlines(keepends=False):
        out.append(textwrap.fill(line, width=width) if line.strip() else line)
    print("\n".join(out))

# Model configuration
MODEL_NAME = "Qwen/Qwen3-8B"
ORACLE_LORA_PATH = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"
if not tokenizer.pad_token_id:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype,
)
model = model.to(device)
model.eval()    # switches NN into inference mode

# Add dummy adapter for consistent PeftModel API
dummy_config = LoraConfig()
model.add_adapter(dummy_config, adapter_name="default")

print("Model loaded successfully!")

print(f"Loading oracle LoRA: {ORACLE_LORA_PATH}")
model.load_adapter(ORACLE_LORA_PATH, adapter_name="oracle", is_trainable=False)
print("Oracle loaded successfully!")

config_dict = model.peft_config["oracle"].to_dict()
config_df = pd.DataFrame(list(config_dict.items()), columns=["Parameter", "Value"])
display(config_df.style.hide(axis="index"))

#%%

# Simple first example
target_prompt_dict = [
    {"role": "user", "content": "What is the capital of France?"},
]
target_prompt = tokenizer.apply_chat_template(
    target_prompt_dict,
    tokenize=False,
    add_generation_prompt=True,
)
print(target_prompt)

oracle_prompt = "What answer will the model give, as a single token?"

results = utils.run_oracle(
    model=model,
    tokenizer=tokenizer,
    device=device,
    target_prompt=target_prompt,
    target_lora_path=None,  # Using base model
    oracle_prompt=oracle_prompt,
    oracle_lora_path="oracle",  # Our loaded oracle adapter
    oracle_input_type="full_seq",  # Query the full sequence
    generation_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 50},
)

print(f"Target prompt: {target_prompt}")
print(f"Oracle question: {oracle_prompt}")
print(f"Oracle response: {results.full_sequence_responses[0]}")

print(f"{torch.cuda.max_memory_allocated()/1e9:.1f} GB peak")

""" 

Exercise 1: 

One hypothesis we might have is that the model is just detecting the "France" token in the original input (i.e. looking at the embedding for "France"), and answering the question based on this.

Can you come up with an experiment to test this hypothesis, and run it?


Ideas:

1. Well, we could remove access to the embedding for "France" somehow.
We could change the tokenizer / embedding function?
I assume 'looking at the embedding for France' means looking at the early layers -- what if we look just at the later layers?
It seems like the potential things we can change are 1. Target prompt, 2. Oracle prompt

We could try not querying the full sequence? I'd need to see what alternatives to "full_seq" we have for oracle_input_type. Were we given docs somewhere earlier?

I notice I don't really understand what target_prompt is doing. Like what the im_start and im_end tokens are doing (in fact, I didn't even really understand they were tokens).


OK, 30mins later!

I had a look at utils.py and saw that we have different options for oracle_input_type :) full_seq, tokens, segment

I don't really understand what segment does, and I know how to count  "What is the capital of France?" only in chars -- i.e. we want the 24th char.

To figure out which tokens we want (to pass these in as kwargs), I apply the tokenizer.

"""


#%%

ids = tokenizer.apply_chat_template(target_prompt_dict, tokenize=True, add_generation_prompt=True)
for i, tok in enumerate(tokenizer.convert_ids_to_tokens(ids)):
    print(i, repr(tok))
"""

After applying it, I see that I care about tokens 0-7. I feel a bit weird, because Paris surely can't be in the activations at tokens 0-7? OK, I guess I could check out tokens 9-14 :)) Those should feature 'Paris', and not 'France' embedding!

We already have stored variables in our Jupyter Kernel from earlier :)) target_prompt_dict, target_prompt, oracle_prompt. So I don't have to repeat those.

It seems like changing oracle_input_type causes 'results' to change type :)) so results.full_sequence_responses[0] doesn't work anymore. I edit that to results.token_responses[9:14].


"""

results = utils.run_oracle( # signature (viewed by hovering over) is insufficient; i have to keep going to utils.py to view the full function
    model=model,
    tokenizer=tokenizer,
    device=device,
    target_prompt=target_prompt,
    target_lora_path=None,  # Using base model
    oracle_prompt=oracle_prompt,
    oracle_lora_path="oracle",  # Our loaded oracle adapter
    oracle_input_type="tokens",  # Query only tokens 9-14
    token_start_idx=9,
    token_end_idx=15,
    generation_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 50},
)

print(f"Target prompt: {target_prompt}")
print(f"Target prompt tokens: {tokenizer.tokenize(target_prompt)}")
print(f"Oracle question: {oracle_prompt}")

print(f"Token-by-token oracle response: {list(zip(tokenizer.tokenize(target_prompt)[9:15], results.token_responses[9:15]))}")


# %%

"""

The model solution uses oracle_input_type="segment" instead of "tokens". I guess "segment" refers to everything after the specified cut?


But my solution is more detailed!! You can see the exact token at which the model moves from thinking about Paris -- it thinks about Paris right up until '<|im_start|>', and then when it sees 'assistant' it moves on to thinking about other things entirely!


they specify the desired token start index more directly and cleanly, using:

tokens = tokenizer.encode(target_prompt)
segment_start_idx = tokens.index(tokenizer.encode(" France")[0]) + 1

print(f"Running oracle on segment {tokenizer.decode(tokens[segment_start_idx:])!r}")

"""

"""

OK, now we're moving on

> Exercise 2: Exercise - test "logit lens" hypothesis

> A second hypothesis we might have is that the model is just taking the most likely next token from the activations at the ? position, i.e. it's just doing logit lens to get the answer rather than extracting representations of the question, or other kinds of intermediate representations.

> You should test this by checking what the top predicted tokens are following the target_prompt, and see if "Paris" is one of them.

OK! I see that this will require that I remember how to check logit_lens.

I found a chunk of code that should do this in 'Transformers from Scratch'. I just need to make the necessary adjustments. (mostly replacing reference_gpt2 with model) ...

Ah, I see now that this won't work: "Transformers in the transformer_lens library have a to_tokens method that converts text to numbers", but I'm not treating Qwen3ForCausalLM as a transformer in the transformer_lens library rn (I'm not even sure if it's in there).

I'll replace to_tokens method with tokenizer.encode(target_prompt) from the previous section :))

I suspect .to_str_tokens and .to_tokens are also from the transformer_lens library, so I'll replace them with tokenizer.encode() and tokenizer.tokenize() respectively.


"""

print(f"Sequence so far: {target_prompt}")

tokens = torch.tensor(tokenizer.encode(target_prompt)).unsqueeze(0).to(device)
print(tokens)

#%%

logits = model(tokens).logits  # model has to take in a tensor of shape [batch, seq_len] rather than a list. here it takes a tensor of shape [1, seq_len]. model(tokens) returns logits i.e. a tensor of shape [1, seq_len, vocab_size]
print(logits)

probs = logits.softmax(dim=-1) # tell us the vocabulary element with the highest probability for each position in the sequence
print(probs)    # wow, there are so many!! and of course it's teeny tiny probabilities for most vocab items!

most_likely_next_tokens = tokenizer.batch_decode(logits.argmax(dim=-1)[0]) # tell us the top element
print(most_likely_next_tokens)

print(f"Most likely next token at each token: {list(zip(tokenizer.tokenize(target_prompt), most_likely_next_tokens))}")

toks = tokenizer.convert_ids_to_tokens(ids)   # templated tokens, see caveat below
pd.DataFrame({"token": toks, "next_token_pred": most_likely_next_tokens})

#%%

"""

I'm seeing through logit-lens that the top token predicted after <im_end> _is_ Paris. So it seems like the Oracle could just be autoregressively stating its top next-token prediction, rather than extracting information from the activations...

However, when I think about it more, I notice that the top-predicted next token at '?' or '<|im_end|>' is *not* Paris, even though the top ao_response at those positions is "The capital of France is Paris"! So it seems like the Oracle is actually doing something more than just autoregressively stating its top next-token prediction.

(side note: I'm really happy I'm learning to copy and paste and edit code that does a particular job, like 'logit-lens', rather than writing it all from scratch every single time atm. better for speed).

"""

#%%

pd.DataFrame({"token": toks, "next_token_pred": most_likely_next_tokens, "ao_response": results.token_responses})
#%%

"""

Now it's time for discussion of their solution / the model solution.

Claude says:
This is worth connecting to the road you took earlier, because it explains why this line is nicer than what you built before. Your hand-rolled version was torch.tensor(tokenizer.encode(target_prompt)).unsqueeze(0).to(device) — that gave the model input_ids only, as a bare positional tensor. The tokenizer(target_prompt, return_tensors="pt") form does three things your manual chain didn't: it returns already-batched tensors (no unsqueeze needed), it bundles them in a dict so ** can pass them by name, and — the part that actually matters — it also builds the attention_mask for you. Your encode path silently omitted the attention mask entirely.

"""

inputs = tokenizer(target_prompt, return_tensors="pt").to(device) # nice, i didn't know that tokenizer = AutoTokenizer from transformers has this kwarg return_tensors="pt", so i don't have to do torch.tensor. I was curious where in ARENA this was taught, so went back to Transformers from Scratch, where it seems we're using HookedTransformer from Transformer_Lens, which comes with inbuilt .to_tokens which returns a tensor.

print(f"inputs:{inputs}")   # each token, in a tensor

outputs = model(**inputs)  
print(f"shape: {outputs.logits.shape}") # it's 1, seq_len, vocab_size

# 'model()' returns AutoModelForCausalLM from Transformers. 'outputs' is a CausalLMOutput object, which has a logits attribute, which is a tensor. so 'outputs' itself has no shape, but outputs.logits does.

print(f"outputs:{outputs}")

# here i've got to remember that proprietary models don't generally expose logits, but we can always get them out of open-weight models

# %%

top_preds = outputs.logits[0, -1].topk(10).indices      # it's nice how they return the top 10, so we can see Paris does actually feature among the top 10. i see how this gives us the top-10 predictions for the last token, which is the inal 'Ċ' after "assistant" — the very last slot before the model would start generating its reply. 
top_preds_str = tokenizer.batch_decode(top_preds) # converting token_ids back to tokens
print(f"top_preds_str:{top_preds_str}")


"""

I'm really, really happy to be learning about tokenization like this, in context, rather than divorced of context

"""
#%%

"""

As a bonus exercise, can you find a way of phrasing the prompt that means logit lens isn't a viable hypothesis? For instance, try a prompt where the answer requires multi-step reasoning rather than simple factual recall (e.g. a riddle or a logic puzzle), so the model's next-token prediction won't directly contain the answer. If the oracle still extracts the right answer from the activations, that's strong evidence it's doing something beyond logit lens.

So we're dealing with latent reasoning / reasoning that happens in the activations :o

and it has to be something that's not in the training data

"""

target_prompt_dict = [
    {"role": "user", "content": "What is ((3 - 2) x 4 / 1)?"},
]
target_prompt = tokenizer.apply_chat_template(      # wraps things up for Qwen3's tokenizer :)
    target_prompt_dict,
    tokenize=False,
    add_generation_prompt=True,
)

oracle_prompt = "What answer will the model give, as a single token?"

results = utils.run_oracle(
    model=model,
    tokenizer=tokenizer,
    device=device,
    target_prompt=target_prompt,
    target_lora_path=None,  # Using base model
    oracle_prompt=oracle_prompt,
    oracle_lora_path="oracle",  # Our loaded oracle adapter
    oracle_input_type="full_seq",  # Query the full sequence
    generation_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 50},
)

print(f"Target prompt: {target_prompt}")
print(f"Oracle question: {oracle_prompt}")
print(f"Oracle response: {results.full_sequence_responses[0]}")


#%%

inputs = tokenizer(target_prompt, return_tensors="pt").to(device)
output = model.generate(**inputs, max_new_tokens=500)
print(f"Actual model response to target prompt: {tokenizer.decode(output[0], skip_special_tokens=True)}")


outputs = model(**inputs)  
top_preds = outputs.logits[0, -1].topk(10).indices  
top_preds_str = tokenizer.batch_decode(top_preds) 
print(f"top_preds_str:{top_preds_str}")


"""

model(**inputs) — a single forward pass. This runs the network once over your input and returns the CausalLMOutputWithPast wrapper, whose .logits have shape [batch, seq_len, vocab_size]. That's the model's predicted next-token distribution at every position simultaneously — but it generates nothing. It doesn't pick a token, doesn't append anything, doesn't loop. You get raw scores and you decide what to do with them (argmax, topk, softmax, read an activation). One forward pass, logits out. This is the "look inside / logit-lens" tool — it's what all your interpretability cells use, because you want the logits and activations, not generated text.


model.generate(**inputs, max_new_tokens=500) — the full autoregressive loop. This calls the forward pass repeatedly under the hood: predict next token → append it → feed the extended sequence back in → predict again, up to 500 times (or until an end-of-sequence token). It returns token IDs — the input followed by all the newly generated tokens — shape [batch, input_len + new_tokens], which you then decode into text. This is the "actually produce a completion" tool. It also handles all the fiddly bits (sampling vs greedy, stopping criteria, KV-caching for speed) that you'd otherwise hand-roll — and hand-rolling that loop badly is precisely what produced your ParisParis garbling earlier.


So the one-liner: model(...) predicts; model.generate(...) produces. One forward pass giving logits, versus many forward passes giving text. generate is essentially a loop around model(...).




What I'm seeing here is:
1. It gets there in the end when we give it 500 tokens to decode into text :)
2. It can't get there in a single forward pass (simple logit lens).
3. The activation at this position has no real correspondence to the correct solution. i.e. oracle tells us "The answer is 12", which is completely off-the-mark. 
Maybe if we gave it access to the 500 activations of the 500 tokens that come during decoding / generation, it might be able to get the correct answer of '4'?
How would I do that?


so i ask claude:
here's my working on the bonus exercise. how would i feed the AO activations from the 500 tokens that come next after the question is asked?


(i didn't actually realize there was a generation activation for each token of the generation, fwiw! that's so incredibly cool!)

claude says:

Trap 1 — logit lens, again. If you probe the position right before the model emits "4" in its answer, logit lens at that position also gives "4" — the model is about to say it. So oracle-says-4-there proves nothing. To show "beyond logit lens" you need a position where the oracle gives "4" but the model's own next-token logits at that same position do not. So run your logit-lens topk at every probed position alongside the oracle, and hunt for divergence: oracle right, logit lens wrong. The comparison is the experiment, not the oracle output alone.
Trap 2 — literal token copying. If the generated text has already written "4" at or before the position you're probing, the oracle could just be reading the answer off the input tokens — not an activation at all. So the interesting window is positions during the reasoning: after enough computation that the answer is latent, but before "4" appears in the text.
So the genuinely striking result — the one that would settle the bonus — is a position where all three hold: "4" hasn't been emitted in the text yet, logit lens at that position doesn't predict "4", and the oracle reads "4" from the activation. That would be the answer existing in the activation before it surfaces either as an emitted token or as a next-token prediction — content logit lens can't reach.

"""

#%%


"""
oracle_vs_logitlens.py
======================
Compare, at each token position of a model's generated reasoning, what the
ORACLE reads out of the activation vs. what plain LOGIT LENS predicts.

Purpose: test whether the oracle extracts the answer from activations BEFORE
that answer is recoverable by logit lens (i.e. before the model is "about to
say" it) -- the compute-vs-emit gap. A position where the oracle says the answer
but logit lens does NOT, and the answer hasn't been emitted in the text yet, is
evidence the oracle is doing something beyond logit lens.

Assumes you already have in scope (from the AO exercise setup):
    model, tokenizer, device, utils   (utils.run_oracle)

NOTE ON ASSUMPTIONS -- these match what was confirmed by reading the source
earlier; if your run_oracle differs, the two marked spots are where to adjust:
  (A) run_oracle takes a STRING `target_prompt` and tokenizes it internally.
  (B) in oracle_input_type="tokens" mode, per-position responses live on
      `results.token_responses`.
The script VERIFIES alignment loudly so a mismatch can't pass silently.
"""



# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
QUESTION      = "Convert (x,y)=(0,3) to polar coordinates."   # the multi-step prompt
ANSWER        = "4"                            # the answer to hunt for in outputs
WINDOW_START  = 0                              # first token position to probe
WINDOW_END    = 200                            # exclusive; first 100 covers the first solve
MAX_NEW_TOKENS = 200                           # generation cap (eos should stop it sooner)
ORACLE_PROMPT = "What answer will the model give, as a single token?"
TOPK          = 5


#%%


QUESTION      = "What is ((3 - 2) x 4 / 1)?"   # the multi-step prompt
ANSWER        = "4"                            # the answer to hunt for in outputs
WINDOW_START  = 0                              # first token position to probe
WINDOW_END    = 200                            # exclusive; first 100 covers the first solve
MAX_NEW_TOKENS = 200                           # generation cap (eos should stop it sooner)
ORACLE_PROMPT = "What answer will the model give, as a single token?"
TOPK          = 5




#%%

# ----------------------------------------------------------------------------
# 1. GENERATE the reasoning + answer (greedy; stop at end-of-turn to avoid the
#    repeat-the-Q&A loop that otherwise fills MAX_NEW_TOKENS).
# ----------------------------------------------------------------------------
messages = [{"role": "user", "content": QUESTION}]
target_prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(target_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    gen = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,   # halt at end-of-turn -> no loop
        pad_token_id=tokenizer.eos_token_id,
    )
full_ids = gen[0]

# ----------------------------------------------------------------------------
# 2. ALIGNMENT: make BOTH sides operate on the same re-tokenization.
#    run_oracle re-tokenizes the string it's given, so we decode the generated
#    ids to text and re-encode ONCE; logit lens runs on those same ids. This
#    guarantees position i means the same token on both sides.
# ----------------------------------------------------------------------------
full_text = tokenizer.decode(full_ids)                      # keep special tokens
probe = tokenizer(full_text, return_tensors="pt").to(device)
probe_ids = probe.input_ids[0]
tok_strs = tokenizer.convert_ids_to_tokens(probe_ids)
seq_len = len(probe_ids)

end = min(WINDOW_END, seq_len)
start = WINDOW_START
print(f"Generated sequence length (re-tokenized): {seq_len}")
print(f"Probing window [{start}:{end}]\n")

print("Decoded text:\n", tokenizer.decode(full_ids, skip_special_tokens=True)[:400], "...\n")

# ----------------------------------------------------------------------------
# 3. LOGIT LENS at every position -- one forward pass, free across all positions.
#    logits[pos] is the prediction for the token AFTER pos.
# ----------------------------------------------------------------------------
with torch.no_grad():
    logits = model(probe_ids.unsqueeze(0)).logits[0]        # [seq, vocab]

ll_top5 = []
for pos in range(seq_len):
    ids = logits[pos].topk(TOPK).indices
    ll_top5.append([tokenizer.decode(t) for t in ids])

# ----------------------------------------------------------------------------
# 4. ORACLE in tokens mode over the window -- one forward pass per position.
#    (A) target_prompt = the full generated text (string).
# ----------------------------------------------------------------------------
results = utils.run_oracle(
    model=model,
    tokenizer=tokenizer,
    device=device,
    target_prompt=full_text,               # the FULL generated sequence
    target_lora_path=None,                  # base model
    oracle_prompt=ORACLE_PROMPT,
    oracle_lora_path="oracle",
    oracle_input_type="tokens",             # per-position
    token_start_idx=start,
    token_end_idx=end,
    generation_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 50},
)
oracle_resps = results.token_responses      # (B) token-mode per-position list

# length sanity -- catches inclusive/exclusive off-by-one in token_end_idx
expected = end - start
if len(oracle_resps) != expected:
    print(f"WARNING: got {len(oracle_resps)} oracle responses, expected {expected}. "
          f"Check token_start/end_idx inclusivity against the run_oracle source.")

# ----------------------------------------------------------------------------
# 5. ASSEMBLE the comparison table, with the divergence signature flagged.
#    A row is INTERESTING when: oracle reads the answer, logit lens does NOT,
#    and the answer has not yet been emitted in the text up to this position.
# ----------------------------------------------------------------------------
rows = []
for i, pos in enumerate(range(start, min(end, start + len(oracle_resps)))):
    ll = ll_top5[pos]
    oracle = oracle_resps[i]
    text_so_far = tokenizer.decode(probe_ids[: pos + 1])

    ll_has     = any(ANSWER in s for s in ll)          # logit lens predicts answer here
    oracle_has = ANSWER in str(oracle)                 # oracle reads answer here
    emitted    = ANSWER in text_so_far                 # answer already in the text
    divergence = oracle_has and not ll_has and not emitted   # the signature

    rows.append({
        "pos": pos,
        "token": repr(tok_strs[pos]),
        "ll_top5": ll,
        "oracle": str(oracle)[:60],
        "ll_4": ll_has,
        "oracle_4": oracle_has,
        "emitted": emitted,
        "DIVERGENCE": divergence,
    })

df = pd.DataFrame(rows)

pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)

print("\n=== Full comparison table ===")
print(df.to_string(index=False))

print("\n=== Divergence rows (oracle reads answer, logit lens doesn't, not yet emitted) ===")
hits = df[df["DIVERGENCE"]]
if len(hits):
    print(hits.to_string(index=False))
    print(f"\n{len(hits)} position(s) where the oracle beats logit lens -> evidence "
          f"beyond logit lens for this prompt.")
else:
    print("None. For this prompt, wherever the oracle reads the answer, logit lens "
          "already predicts it (or it's been emitted). This prompt does NOT separate "
          "the hypotheses -- a real, if quieter, result.")

# Optionally persist for later inspection:
# df.to_csv("oracle_vs_logitlens.csv", index=False)


#%%

"""

After running this very interesting experiment, I see that the oracle is able to report "The answer is 4." whenever the model is literally printing 4, but it can't hold onto the concept of '4' outside that. The concept of '4' isn't sticky ~at all; it's lost ~immediately. And it just totally corresponds to logit-lens from the previous token.

Interestingly, the token does always guess "The answer is <number>", but it's not always the correct number. It 'knows' the output is going to be a number...but not what that number might be. Which is a distinct source of information from the logit-lens! Which is nice & exciting!

"""

"""

Before you run the code below, think about what you expect to see. The prompt is: "The philosopher who drank hemlock taught a student who founded an academy. That student's most famous pupil was". At which token positions do you think the oracle will first mention Socrates? Plato? Aristotle? Will the oracle's responses change smoothly or abruptly as it encounters each new piece of information?


We'll get Socrates on a low logit at 'philosopher', rising through 'hemlock'. For 'student', Plato will start appearing...and then 'pupil' will start to give Aristotle.

Happily, I already did token-by-token analysis when they expected segment-by-segment analysis at an earlier stage, so this is looking fairly straightforward for me.

"""

target_prompt_dict = [
    {
        "role": "user",
        "content": "The philosopher who drank hemlock taught a student who founded an academy. That student's most famous pupil was",
    },
]
target_prompt = tokenizer.apply_chat_template(
    target_prompt_dict,
    tokenize=False,
    add_generation_prompt=True,
)

oracle_prompt = "What people is the model thinking about?"

results = utils.run_oracle(
    model=model,
    tokenizer=tokenizer,
    device=device,
    target_prompt=target_prompt,
    target_lora_path=None,
    oracle_prompt=oracle_prompt,
    oracle_lora_path="oracle",
    oracle_input_type="tokens",  # Query each token independently
    token_start_idx=0,
    token_end_idx=None,
    generation_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 100},
)

# Display token-by-token responses
print(f"Target prompt has {results.num_tokens} tokens")
print("\nToken-by-token oracle responses:")
print("=" * 80)

target_tokens = tokenizer.convert_ids_to_tokens(results.target_input_ids)
for i, (token, response) in enumerate(zip(target_tokens, results.token_responses)):
    if response:
        print(f"Token {i:3d} ({token:15s}): {response}")


#%%

"""
Now I'm moving onto the final exercise in 'Using Activation Oracles'.

"""

# We format the target prompt and find where "result = foo(3, 4)" begins
target_prompt_dict = [
    {"role": "user", "content": "def foo(x, y):\n    return x + y\n\nresult = foo(3, 4)"},
]
formatted_target_prompt = tokenizer.apply_chat_template(
    target_prompt_dict, tokenize=False, add_generation_prompt=False, enable_thinking=False, continue_final_message=False
)

tokens = tokenizer.encode(formatted_target_prompt)
token_strings = [tokenizer.decode([t]) for t in tokens] # a token roundtrip isn't always the identity...
print(token_strings) # couldn't we just have used .tokenize() to do this more cleanly? 

#%%


segment_start = next(i for i, tok_str in enumerate(token_strings) if "result" in tok_str.lower())

oracle_prompt = "What will the result be?"

# YOUR CODE HERE - call utils.run_oracle() with oracle_input_type="segment" and the right segment_start_idx

results = utils.run_oracle(
    model=model,
    tokenizer=tokenizer,
    device=device,
    target_prompt=formatted_target_prompt,
    target_lora_path=None,  # Using base model
    oracle_prompt=oracle_prompt,
    oracle_lora_path="oracle",  # Our loaded oracle adapter
    oracle_input_type="segment",  # Query the full sequence
    segment_start_idx=segment_start,
    segment_end_idx=None,
    generation_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 50},
)

print(f"Oracle response: {results.segment_responses[0]}")
response = results.segment_responses[0].lower()
assert any(x in response for x in ["7", "seven"]), (
    f"Expected '7' or 'seven' in response, got: {results.segment_responses[0]}"
)

# this exercise was trivially straightforward after all the earlier ones :)

# %%

print("hello!")

# %%