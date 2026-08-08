package app.presence.onnx

import app.presence.audio.FbankExtractor
import app.presence.audio.SAMPLE_RATE_HZ
import app.presence.audio.millisToSamples
import app.presence.engine.Embedding
import app.presence.engine.SpeakerEmbedder

/**
 * A speaker-embedding model via ONNX Runtime.
 *
 * The shipped model's verified signature is `feats [B, T, 80] -> embs [B, 256]`: it consumes
 * log-mel filterbank features, not raw audio, so [FbankExtractor] runs first. Tensor names stay
 * constructor parameters because a fine-tuned replacement (spec §4.1, L3) may export different
 * ones.
 */
class OnnxSpeakerEmbedder(
    private val session: OnnxModelSession,
    private val fbank: FbankExtractor = FbankExtractor(),
    private val inputName: String = "feats",
    private val outputName: String = "embs",
) : SpeakerEmbedder, AutoCloseable {

    private companion object {
        /** Below this the embedding is dominated by noise (spec §4.1). */
        const val MIN_DURATION_MS = 400L
    }

    private var discoveredDimension: Int = 0

    override val dimension: Int
        get() {
            check(discoveredDimension > 0) { "dimension is unknown until embed() has been called once" }
            return discoveredDimension
        }

    override fun embed(samples: FloatArray): Embedding {
        val minimumSamples = millisToSamples(MIN_DURATION_MS)
        require(samples.size >= minimumSamples) {
            "need at least $MIN_DURATION_MS ms ($minimumSamples samples) at $SAMPLE_RATE_HZ Hz, " +
                "got ${samples.size}"
        }

        val features = fbank.extract(samples)
        val frames = features.size
        val bins = features[0].size

        // Flatten [T][80] into the [1, T, 80] tensor the model declares.
        val flat = FloatArray(frames * bins)
        for (frame in 0 until frames) {
            features[frame].copyInto(flat, frame * bins)
        }
        val tensor = session.floatTensor(flat, longArrayOf(1, frames.toLong(), bins.toLong()))

        return session.runTensors(mapOf(inputName to tensor)) { result ->
            @Suppress("UNCHECKED_CAST")
            val raw = (result.get(outputName).get().value as Array<FloatArray>)[0]
            discoveredDimension = raw.size
            // Normalizing here means every downstream cosine is a plain dot product.
            Embedding.ofNormalized(raw)
        }
    }

    override fun close() {
        session.close()
    }
}
