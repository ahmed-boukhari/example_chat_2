const return_timestamps = kwargs.return_timestamps ?? false;
    const chunk_length_s = kwargs.chunk_length_s ?? 0;
    const force_full_sequences = kwargs.force_full_sequences ?? false;
    let stride_length_s = kwargs.stride_length_s ?? null;

    const generation_config = { ...kwargs }

    if (return_timestamps === 'word') {
        generation_config['return_token_timestamps'] = true;
        generation_config['return_timestamps'] = false; // Do not predict timestamp tokens
    }

    const single = !Array.isArray(audio);
    if (single) {
        audio = [/** @type {AudioInput} */ (audio)];
    }

    // @ts-expect-error TS2339
    const time_precision = this.processor.feature_extractor.config.chunk_length / this.model.config.max_source_positions;
    const hop_length = this.processor.feature_extractor.config.hop_length;

    const sampling_rate = this.processor.feature_extractor.config.sampling_rate;
    const preparedAudios = await prepareAudios(audio, sampling_rate);

    const toReturn = [];
    for (const aud of preparedAudios) {
        /** @type {{stride: number[], input_features: Tensor, is_last: boolean, tokens?: bigint[], token_timestamps?: number[]}[]} */
        let chunks = [];
        if (chunk_length_s > 0) {
            if (stride_length_s === null) {
                stride_length_s = chunk_length_s / 6;
            } else if (chunk_length_s <= stride_length_s) {
                throw Error("`chunk_length_s` must be larger than `stride_length_s`.")
            }

            // TODO support different stride_length_s (for left and right)

            const window = sampling_rate * chunk_length_s;
            const stride = sampling_rate * stride_length_s;
            const jump = window - 2 * stride;
            let offset = 0;

            // Create subarrays of audio with overlaps
            while (true) {
                const offset_end = offset + window;
                const subarr = aud.subarray(offset, offset_end);
                const feature = await processor(subarr);

                const is_first = offset === 0;
                const is_last = offset_end >= aud.length;
                chunks.push({
                    stride: [
                        subarr.length,
                        is_first ? 0 : stride,
                        is_last ? 0 : stride
                    ],
                    input_features: feature.input_features,
                    is_last,
                })
                if (is_last) break;
                offset += jump;
            }

        } else {
            chunks = [{
                stride: [aud.length, 0, 0],
                input_features: (await processor(aud)).input_features,
                is_last: true
            }]
        }

        // Generate for each set of input features
        for (const chunk of chunks) {
            generation_config.num_frames = Math.floor(chunk.stride[0] / hop_length);

            // NOTE: doing sequentially for now
            const data = await generate({
                inputs: chunk.input_features,
                ...generation_config
            });

            // TODO: Right now we only get top beam
            if (return_timestamps === 'word') {
                // @ts-expect-error TS2339
                chunk.tokens = data.sequences.tolist()[0];
                // @ts-expect-error TS2339
                chunk.token_timestamps = data.token_timestamps.tolist()[0].map(
                    (/** @type {number} */ x) => round(x, 2)
                );

            } else {
                chunk.tokens = (/** @type {Tensor} */(data))[0].tolist();
            }

            // convert stride to seconds
            chunk.stride = chunk.stride.map(x => x / sampling_rate);
        }

        // Merge text chunks
        // @ts-ignore
        const [full_text, optional] = this.tokenizer._decode_asr(chunks, {
            time_precision, return_timestamps, force_full_sequences
        });

        toReturn.push({ text: full_text, ...optional })
    }
    return single ? toReturn[0] : toReturn;