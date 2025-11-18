import {
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    StoppingCriteria,
    pipeline,
    env
} from './transformers.js';

// ============================================================================
// ENVIRONMENT CONFIGURATION
// ============================================================================

env.localModelPath = './models/';
env.allowRemoteModels = false;
env.allowLocalModels = true;
var loc = location.pathname;
var dir = loc.substring(0, loc.lastIndexOf('/'));
env.backends.onnx.wasm.wasmPaths = decodeURIComponent(dir) + '/';
console.log('[Worker] WASM paths set to:', decodeURIComponent(dir));
// env.backends.onnx.wasm.numThreads = 4;
// ============================================================================
// STREAMING & INTERRUPTION UTILITIES
// ============================================================================

class StreamingTextCallback extends TextStreamer {
    constructor(tokenizer, onTextGenerated) {
        super(tokenizer, {
            skip_prompt: true,
            skip_special_tokens: true,
        });
        console.log('[StreamingTextCallback] Initialized with tokenizer:', tokenizer.config.tokenizer_class);
        this.onTextGenerated = onTextGenerated;
    }

    on_finalized_text(text) {
        this.onTextGenerated(text);
    }
}

class InterruptibleGeneration extends StoppingCriteria {
    constructor() {
        super();
        this.shouldStop = false;
    }

    interrupt() {
        this.shouldStop = true;
    }

    reset() {
        this.shouldStop = false;
    }

    _call(input_ids, scores) {
        return new Array(input_ids.length).fill(this.shouldStop);
    }
}

// ============================================================================
// CHAT MODEL FACTORY (Singleton)
// ============================================================================

class ChatModelFactory {
    static model = null;
    static tokenizer = null;
    static modelId = null;
    static dtype = null;

    static async getInstance(progressCallback = null) {
        this.tokenizer ??= AutoTokenizer.from_pretrained(this.modelId, {
            progress_callback: progressCallback,
        });
        this.tokenizer.then(tokenizer => {
            console.log('[ChatModel] Tokenizer loaded:', tokenizer.config.tokenizer_class);
        });

        this.model ??= AutoModelForCausalLM.from_pretrained(this.modelId, {
            dtype: this.dtype,
            device: "wasm",
            progress_callback: progressCallback,
        });

        return Promise.all([this.tokenizer, this.model]);
    }

    static dispose() {
        this.model = null;
        this.tokenizer = null;
        this.modelId = null;
        this.dtype = null;
    }
}

// ============================================================================
// SPEECH RECOGNITION MODEL FACTORY (Singleton)
// ============================================================================

class SpeechRecognitionModelFactory {
    static taskType = "automatic-speech-recognition";
    static modelId = null;
    static pipelineInstance = null;

    static async getInstance(progressCallback = null) {
        if (this.pipelineInstance === null) {
            const options = {
                quantized: null,
                progress_callback: progressCallback,
            };

            // Whisper medium models need special revision to avoid OOM
            if (this.modelId.includes("/whisper-medium")) {
                options.revision = "no_attentions";
            }
            console.log(`[SpeechRecognitionModelFactory] Creating pipeline with options:`, JSON.stringify(options));
            console.log(`[SpeechRecognitionModelFactory] Model ID: ${this.modelId}`);
            this.pipelineInstance = pipeline(this.taskType, this.modelId, options);
        }

        return this.pipelineInstance;
    }

    static async dispose() {
        if (this.pipelineInstance !== null) {
            const instance = await this.pipelineInstance;
            if (instance && typeof instance.dispose === 'function') {
                instance.dispose();
            }
            this.pipelineInstance = null;
        }
        this.modelId = null;
    }
}

// ============================================================================
// SUMMARIZATION MODEL FACTORY (Singleton)
// ============================================================================

class SummarizationModelFactory {
    static taskType = "summarization";
    static modelId = null;
    static isQuantized = null;
    static pipelineInstance = null;

    static async getInstance(progressCallback = null) {
        if (this.pipelineInstance === null) {
            this.pipelineInstance = pipeline(this.taskType, this.modelId, {
                quantized: this.isQuantized,
                progress_callback: progressCallback,
            });
        }

        return this.pipelineInstance;
    }

    static async dispose() {
        if (this.pipelineInstance !== null) {
            const instance = await this.pipelineInstance;
            if (instance && typeof instance.dispose === 'function') {
                instance.dispose();
            }
            this.pipelineInstance = null;
        }
        this.modelId = null;
        this.isQuantized = null;
    }
}

// ============================================================================
// CHAT GENERATION HANDLERS
// ============================================================================

const generationInterrupter = new InterruptibleGeneration();

async function loadChatModel(modelId, dataType) {
    self.postMessage({
        status: 'loading',
        task: 'chat-generation',
        data: 'Loading chat model...'
    });
    
    ChatModelFactory.modelId ??= modelId;
    ChatModelFactory.dtype ??= dataType;
    
    const [tokenizer, model] = await ChatModelFactory.getInstance(progressData => {
        self.postMessage(progressData);
    });

    self.postMessage({
        status: 'loading',
        task: 'chat-generation',
        data: 'Compiling shaders and warming up model...'
    });

    // Warmup run with dummy input
    const warmupInputs = tokenizer('a');
    await model.generate({ ...warmupInputs, max_new_tokens: 1 });
    
    self.postMessage({ status: 'ready',
        task: 'chat-generation'
     });
}

async function generateChatResponse(messages) {
    const [tokenizer, model] = await ChatModelFactory.getInstance();

    const modelInputs = tokenizer.apply_chat_template(messages, {
        add_generation_prompt: true,
        return_dict: true,
    });

    let generationStartTime;
    let tokensGenerated = 0;
    
    const onTokenGenerated = (text) => {
        generationStartTime ??= performance.now();

        let tokensPerSecond;
        if (tokensGenerated++ > 0) {
            const elapsedMs = performance.now() - generationStartTime;
            tokensPerSecond = tokensGenerated / elapsedMs * 1000;
        }
        
        self.postMessage({
            status: 'update',
            task: 'chat-generation',
            output: text,
            tps: tokensPerSecond,
            numTokens: tokensGenerated,
        });
    }

    const textStreamer = new StreamingTextCallback(tokenizer, onTokenGenerated);
    self.postMessage({ status: 'start',
        task: 'chat-generation',
     });

    const generatedOutputs = await model.generate({
        ...modelInputs,
        max_new_tokens: 512,
        streamer: textStreamer,
        stopping_criteria: generationInterrupter,
    });
    console.log('[Generation] Completed with outputs:', generatedOutputs);

    const decodedText = tokenizer.batch_decode(generatedOutputs, { 
        skip_special_tokens: true ,
        clean_up_tokenization_spaces: true,
    });
    console.log('[Generation] Decoded text:', decodedText);

    self.postMessage({
        status: 'complete',
        task: 'chat-generation',
        output: decodedText,
    });
}

function interruptGeneration() {
    generationInterrupter.interrupt();
}

function resetGeneration() {
    generationInterrupter.reset();
}

// ============================================================================
// SPEECH RECOGNITION HANDLERS
// ============================================================================

async function transcribeAudio(audioData, modelName, isMultilingual, useQuantization, taskType, languageCode) {
    const isDistilWhisper = modelName.startsWith("distil-whisper/");

    // Build full model name
    let fullModelName = modelName;
    // if (!isDistilWhisper && !isMultilingual) {
    //     fullModelName += ".en";
    // }

    const factory = SpeechRecognitionModelFactory;
    
    // Reload model if configuration changed
    const needsReload = factory.modelId !== fullModelName;
    if (needsReload) {
        factory.modelId = fullModelName;
        console.log(`[SpeechRecognition] Loading model: ${fullModelName}`);
    }

    const recognizer = await factory.getInstance(progressData => {
        console.log('[SpeechRecognition] Progress:', JSON.stringify(progressData));
        self.postMessage(progressData);
    });

    const timePrecision =
        recognizer.processor.feature_extractor.config.chunk_length /
        recognizer.model.config.max_source_positions;

    let transcriptionChunks = [{
        tokens: [],
        finalised: false,
    }];

    function onChunkComplete(chunk) {
        const lastChunk = transcriptionChunks[transcriptionChunks.length - 1];
        Object.assign(lastChunk, chunk);
        lastChunk.finalised = true;

        if (!chunk.is_last) {
            transcriptionChunks.push({
                tokens: [],
                finalised: false,
            });
        }
    }

    function onGenerationStep(generationResult) {
        console.log('[SpeechRecognition] Generation step:', JSON.stringify(generationResult));
        const lastChunk = transcriptionChunks[transcriptionChunks.length - 1];
        lastChunk.tokens = [...generationResult[0].output_token_ids];

        const decodedTranscript = recognizer.tokenizer._decode_asr(transcriptionChunks, {
            time_precision: timePrecision,
            return_timestamps: true,
            force_full_sequences: false,
        });
        console.log('[SpeechRecognition] Update:', JSON.stringify(decodedTranscript));
        self.postMessage({
            status: "update",
            task: "automatic-speech-recognition",
            data: decodedTranscript,
        });
    }
    

    const transcriptionResult = await recognizer(audioData, {
        top_k: 0,
        do_sample: false,
        chunk_length_s: isDistilWhisper ? 20 : 30,
        stride_length_s: isDistilWhisper ? 3 : 5,
        language: languageCode,
        task: taskType,
        return_timestamps: true,
        force_full_sequences: false,
        callback_function: onGenerationStep,
        chunk_callback: onChunkComplete,
    }).catch((error) => {
        console.error('[SpeechRecognition] Error:', JSON.stringify(error));
        self.postMessage({
            status: "error",
            task: "automatic-speech-recognition",
            data: error,
        });
        return null;
    });
    console.log('[SpeechRecognition] Final transcription result:', JSON.stringify(transcriptionResult));
    return transcriptionResult;
}

// ============================================================================
// SUMMARIZATION HANDLERS
// ============================================================================

async function summarizeText(inputText, modelName, maximumLength, minimumLength) {
    const factory = SummarizationModelFactory;
    factory.modelId = modelName;
    factory.isQuantized = false;
    const summarizer = await factory.getInstance(progressData => {
        self.postMessage({
            status: "progress",
            task: "summarization",
            data: progressData,
        });
    });

    // Truncate very long inputs
    const maxInputChars = 1024 * 4;
    const truncatedInput = inputText.length > maxInputChars 
        ? inputText.slice(0, maxInputChars) 
        : inputText;

    const summaryResult = await summarizer(truncatedInput, {
        max_length: maximumLength,
        min_length: minimumLength,
        do_sample: false,
    }).catch((error) => {
        self.postMessage({
            status: "error",
            task: "summarization",
            data: error,
        });
        return null;
    });

    return summaryResult;
}

// ============================================================================
// MESSAGE ROUTER
// ============================================================================

self.addEventListener('message', async (event) => {
    const { type, task, data } = event.data;

    // Handle chat generation operations
    if (type) {
        switch (type) {
            case 'load':
                await loadChatModel(data.model_id, data.dtype);
                break;

            case 'generate':
                resetGeneration();
                await generateChatResponse(data);
                break;

            case 'interrupt':
                interruptGeneration();
                break;

            case 'reset':
                resetGeneration();
                break;
        }
        return;
    }

    // Handle pipeline tasks
    if (task === "automatic-speech-recognition") {
        const transcript = await transcribeAudio(
            event.data.audio,
            event.data.model,
            event.data.multilingual,
            event.data.quantized,
            event.data.subtask,
            event.data.language,
        );
        console.log('[Worker] Transcription complete:', JSON.stringify(transcript));
        if (transcript !== null) {
            self.postMessage({
                status: "complete",
                task: "automatic-speech-recognition",
                data: transcript,
            });
        }
    } 
    else if (task === "summarization") {
        const summary = await summarizeText(
            event.data.text,
            event.data.model,
            event.data.maxLength,
            event.data.minLength,
        );
        
        if (summary !== null) {
            self.postMessage({
                status: "complete",
                task: "summarization",
                data: summary,
            });
        }
    }
});