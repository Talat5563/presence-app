package app.presence.harness

import app.presence.audio.SAMPLE_RATE_HZ
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Path
import javax.sound.sampled.AudioInputStream
import javax.sound.sampled.AudioSystem
import javax.sound.sampled.UnsupportedAudioFileException
import kotlin.io.path.exists
import kotlin.io.path.name

/** Thrown when a file cannot be used as-is. The message always says how to fix it. */
class UnsupportedAudioException(message: String) : Exception(message)

/**
 * Reads 16 kHz mono 16-bit PCM WAV into normalized floats.
 *
 * Deliberately **refuses to resample**. Spec §9.1 mandates 16 kHz mono corpora, and silent
 * resampling would corrupt exactly the spectral detail speaker embeddings depend on — a bug
 * that would show up as unexplained accuracy loss rather than as an error.
 */
object WavReader {

    private const val BITS_PER_SAMPLE = 16
    private const val FULL_SCALE = 32_768f

    fun read(path: Path): FloatArray {
        if (!path.exists()) {
            throw UnsupportedAudioException("audio file not found: $path")
        }

        // Everything touching the audio layer sits inside this try. AudioSystem throws checked
        // exceptions on both open and read, and Task 17 walks thousands of corpus files catching
        // UnsupportedAudioException per file — a foreign type escaping here would kill the batch.
        return try {
            AudioSystem.getAudioInputStream(path.toFile()).use { audio -> decode(path, audio) }
        } catch (failure: UnsupportedAudioFileException) {
            throw UnsupportedAudioException("${path.name}: not a readable audio file (${failure.message})")
        } catch (failure: IOException) {
            throw UnsupportedAudioException("${path.name}: could not be read (${failure.message})")
        }
    }

    private fun decode(path: Path, audio: AudioInputStream): FloatArray {
            val format = audio.format

            if (format.sampleRate.toInt() != SAMPLE_RATE_HZ) {
                throw UnsupportedAudioException(
                    "${path.name}: sample rate is ${format.sampleRate.toInt()} Hz, " +
                        "expected $SAMPLE_RATE_HZ Hz. Convert it with:\n" +
                        "  ffmpeg -i ${path.name} -ar $SAMPLE_RATE_HZ -ac 1 -c:a pcm_s16le converted.wav",
                )
            }
            if (format.sampleSizeInBits != BITS_PER_SAMPLE) {
                throw UnsupportedAudioException(
                    "${path.name}: ${format.sampleSizeInBits}-bit samples, expected $BITS_PER_SAMPLE-bit PCM. " +
                        "Convert it with:\n" +
                        "  ffmpeg -i ${path.name} -ar $SAMPLE_RATE_HZ -ac 1 -c:a pcm_s16le converted.wav",
                )
            }

            val channels = format.channels
            val bytes = audio.readAllBytes()
            val order = if (format.isBigEndian) ByteOrder.BIG_ENDIAN else ByteOrder.LITTLE_ENDIAN
            val shorts = ByteBuffer.wrap(bytes).order(order).asShortBuffer()

            if (shorts.limit() % channels != 0) {
                throw UnsupportedAudioException(
                    "${path.name}: ${shorts.limit()} samples do not divide evenly into $channels " +
                        "channels — the file is truncated",
                )
            }

            val frameCount = shorts.limit() / channels
            val samples = FloatArray(frameCount)
            for (frame in 0 until frameCount) {
                var sum = 0f
                for (channel in 0 until channels) {
                    sum += shorts.get(frame * channels + channel) / FULL_SCALE
                }
                // Dividing by 32_768 already bounds this to [-1, 1]; the clamp is belt-and-braces
                // against a malformed frame count rather than the thing enforcing the range.
                samples[frame] = (sum / channels).coerceIn(-1f, 1f)
            }
            return samples
        }
    }
}
