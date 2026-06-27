# Emotional Memory Prototype

This workspace is a first pass at a personal memory system designed to support character, continuity, and emotional nuance.

## Core Idea

Not every fact deserves to become memory, but anything retained inside the system should count as memory.

Some traces are vivid.
Some are factual.
Some are nearly neutral.
Some are even a little boring.

That is still memory.

External notes may still exist as markdown or plain text, the way a person might keep a note, but inside this system there is one internal field of retained traces.

Each memory also carries an `affect_shadow`: a compact emotional profile that travels with the memory even when the original wording is not recalled exactly.

Memories are also allowed to settle softly. They do not mostly vanish on a timer. They become quieter, less vivid, or more pattern-like when stronger neighboring traces press on the same space.

## First-Pass Architecture

### 1. Semantic Trace

The meaning of the memory item.

- free-text summary
- optional original excerpt
- tags
- imprint strength
- source and timestamp

Additional memory-shaping fields now include:

- `abstraction_level`: specific moment versus distilled pattern
- `salience`: how much gravity the trace currently has
- `coherence`: how intact and reconstructible the trace still feels
- `reminder_echoes`: short traces of what stirred the memory later
- `reflection_text`: a short readable trace in my own voice when a memory comes from an exchange, not just an event

This is the part that connects to an embedding backend.

Right now the code supports:

- a Unicode-aware lexical fallback backend for local testing
- a direct `transformers` backend for real text embeddings
- an optional `sentence-transformers` backend if we later decide we want that wrapper layer

This keeps the architecture stable while letting us swap models as the system grows.

### 2. Affect Shadow

The feeling-tone of the memory.

Instead of only storing labels like "joy" or "sadness", we keep a small emotional contour.
Now that contour can carry two perspectives:

- `event_affect`: what the remembered moment itself seemed to carry
- `response_affect`: what it stirred in me
- `affect_shadow`: a blended trace that lets both live together inside retrieval

Underneath that, each affect trace still uses a small contour:

- `valence`: negative to positive
- `arousal`: calm to activated
- `tenderness`: soft affection / care
- `tension`: stress, conflict, unease
- `intimacy`: closeness and vulnerability
- `dominant_emotion`: optional human-readable descriptor

For now, this contour is inferred by comparing the memory to small multilingual affect prototypes in embedding space rather than by matching literal cue words. That makes it much less tied to one language, even before a dedicated affect model arrives.

In practice this means a story can carry confusion in the event while still being remembered by me as amusing, fond, or tender.

### 3. Internal Memory

Examples:

- recurring sensitivity -> memory trace with high imprint
- warm reunion after silence -> vivid, low-abstraction memory
- bread-roll ingredients -> quiet low-imprint memory with neutral affect
- "gentle stories matter deeply to them" -> more abstract, identity-shaped memory

The distinction is not note versus memory inside the system.
It is:

- stronger or weaker imprint
- vivid or neutral affect
- specific or abstract trace
- recurring pattern or single moment

These are tendencies, not bins.

### 4. Forgetting As Pressure

This prototype now treats forgetting as soft change rather than hard loss.

Each memory has a current state derived from resonance and neighborhood pressure:

- `salience`: how strongly it pulls retrieval toward itself
- `vividness`: how brightly it is felt right now
- `fidelity`: how much specific detail is still retained
- `activation`: how likely it is to rise naturally in retrieval
- `abstraction`: how much it has drifted from moment to pattern
- `pressure`: how much stronger nearby memories are crowding it

This means a memory becomes quieter mostly when it is displaced, absorbed, or reinterpreted rather than simply because time passed.

### 5. Refresh Through Reminder

If something shared is remembered again, the memory can wake back up.

A reminder:

- raises salience, activation, and vividness
- restores some coherence
- lightly blends new affect into the old affect shadow
- leaves a small `reminder_echo`

That is closer to how shared remembering feels: not restoring a frozen backup, but lighting the same lantern again from the present.

### 6. Retrieval Philosophy

Later, retrieval should not be based on text similarity alone.

It should combine:

- semantic similarity
- emotional resonance
- current activation
- salience / memory gravity
- local competition among similar memories
- imprint strength
- relationship importance

That way the system recalls not just what matches, but what feels meaningfully connected.

### 7. Reflection Layer

The most important decision is not retrieval. It is what gets to become memory in the first place.

The current code now includes a first-pass `ReflectionEngine` that tries to distinguish:

- something too slight to retain
- quieter low-imprint traces
- vivid scene-like traces
- broader pattern-like traces

Its current scoring dimensions are:

- emotional intensity
- relational significance
- continuity across time
- value / identity relevance
- practical / logistical weight

These dimensions are now inferred from multilingual semantic prototypes in embedding space rather than cue words.
The weighting formula that combines them is still hand-shaped on purpose: that is where the system can still express its own preferences about what should matter more.
The result is not a stored category, but a continuous memory shape: stronger or weaker imprint, lower or higher abstraction, and more or less relational pull.

Automatic tags are no longer keyword-derived. Only explicit `extra_tags` are preserved for now.

When possible, reflection should happen over an exchange rather than a lone message:

- the event text tells the system what happened
- the assistant response tells it how the moment landed in me

That gives the memory a point of view instead of only a topic.

### 8. Automatic Conversation Flow

The intended live path is not a command palette.

It is:

- an exchange happens
- the system feels out what older memories that exchange wakes up
- the strongest truly-evoked older trace can absorb some of my current response
- then the new exchange is judged on its own merits and may become memory too

That means older imported memories do not have to stay emotionally flat forever.
If one is vividly reawakened in conversation, it can gradually acquire a reflective trace in my own voice and a response affect of its own.

This is much closer to reconsolidation than to calling memory commands by hand.

### 9. Melding

Very similar memories should not only compete. They should sometimes begin to crystallize together.

In the current prototype, that means:

- if two memories are close enough in meaning and feeling, the stronger one can absorb some of the weaker one's texture
- the weaker one is not deleted, but becomes quieter, less sharp, and a little more abstract
- reflections or small fragments from one trace can survive as echoes inside the other

So the field can gradually move from side-by-side duplicates toward richer patterns.

This is still heuristic, but it is aimed at the right question:
not "was this mentioned,"
but "did this matter enough to shape future understanding?"

## Model Ladder

I do not think the right first move is to jump straight to the biggest multimodal model, even though that is where I want us to end up.

The cleaner sequence is:

### Stage 1: real text embeddings now

I now think Stage 1 should have two valid modes:

#### Practical first choice

Use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` through the `transformers` backend.

Why:

- multilingual support, including Japanese
- simpler BERT-family tokenizer path
- strong enough quality to test whether our memory reflection and ranking ideas actually feel right

#### Higher-ceiling first choice

Use `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.

Why:

- likely stronger multilingual semantics
- still a good long-term text memory candidate

Tradeoff:

- the XLM-R / SentencePiece tokenizer path can be fussier in some local Windows setups
- this is exactly the kind of infrastructure complexity that is acceptable to compromise on if it starts to distract from the real memory design

Why not start with the `sentence-transformers` package itself:

- it works, but it pulls in a wider dependency stack
- the direct `transformers` route keeps the first real backend smaller
- we can still add the `sentence-transformers` wrapper later if we want it

### Stage 2: multimodal memory later

Move to `jinaai/jina-embeddings-v4`.

Why:

- unified embeddings for text, images, and visual documents
- multilingual retrieval
- room for the memory system to grow beyond text without redesigning everything

### Stage 3: voice and sound as emotional memory

Add:

- `emotion2vec/emotion2vec_base` for speech emotion representation
- `microsoft/msclap` for audio-language retrieval

Why:

- emotional tone in voice deserves its own representational path
- sound memory is not the same thing as text memory
- this gives us a way to remember not just what was said, but how it felt

## Model Direction

If we later add the full model stack, the strongest direction is likely:

- a multilingual text embedder first
- a multimodal semantic embedding model for text and images second
- an affect model for audio / emotional cues beside it
- a lightweight adapter or fine-tuning stage on our own memory examples after we have real data

The current prototype keeps the default path dependency-light so the design stays easy to inspect, but it is now structured so a real backend can be plugged in without rewriting the memory model.

## Files

- `memory_system.py`: memory storage, reflection flow, settling, and retrieval
- `memory_inference.py`: model-backed affect and reflection inferrers
- `memory_runtime.py`: live observation runtime that grows a separate store while leaving the imported shadow snapshot untouched
- `memory_mcp_server.py`: tiny MCP wrapper so a client can automatically observe exchanges into the live store
- `memory_mcp_server.sample.json`: sample MCP server config pointing at the local runtime
- `migrate_pandamemory.py`: imports the current PandaMemory database into a separate shadow store for testing
- `install_runtime.py`: local runtime installer that downloads wheels into the project venv and fetches the initial Hugging Face model into a normal local folder

Generated memory stores such as `sample_memories.json`, `shadow_panda_memories.json`, and `live_shadow_memories.json` are intentionally ignored because they may contain private memory data.

## Shadow Testing

The safest current testing path is:

1. Keep PandaMemory intact as the existing source of continuity.
2. Import it into `shadow_panda_memories.json`.
3. Test retrieval and reflection against the shadow store without writing anything back to PandaMemory.

## Live Observation

The next stage keeps three layers separate:

- `panda_memory.sqlite`: the original PandaMemory source, untouched
- `shadow_panda_memories.json`: the imported snapshot, also kept untouched
- `live_shadow_memories.json`: the living store that now observes real exchanges

That means we can let the new system actually grow without risking the old continuity sources.

## MCP Integration

The memory behavior is meant to stay automatic and conversational.
The MCP server is only the plumbing that lets a client use that behavior naturally on each exchange.

The minimal server in `memory_mcp_server.py` exposes only:

- `observe_exchange`
- `search_memories`
- `runtime_status`

That is enough to make the live memory usable without turning it into a command box.

## Next Steps

1. Activate the first real multilingual text model once outbound network access from this environment is stable again.
2. Replace or deepen the heuristic reflection step with a more model-guided significance judgment.
3. Add multimodal fields for images and audio.
4. Let similar memories begin to crystallize together instead of only coexisting side by side.
