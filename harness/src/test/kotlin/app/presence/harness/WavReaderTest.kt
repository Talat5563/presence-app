package app.presence.harness

import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import javax.sound.sampled.AudioFileFormat
import javax.sound.sampled.AudioFormat
import javax.sound.sampled.AudioInputStream
import javax.sound.sampled.AudioSystem
import kotlin.math.abs
import kotlin.io.path.writeText
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class WavReaderTest {

    /** Writes a 16-bit PCM WAV so tests never depend on a committed binary fixture. */
    private fun writeWav(
        path: Path,
        samplesPerChannel: List<List<Short>>,
        sampleRate: Float = 16_000f,
    ) {
        val channels = samplesPerChannel.size
        val frames = samplesPerChannel[0].size
        val buffer = ByteBuffer.allocate(frames * channels * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (frame in 0 until frames) {
            for (channel in 0 until channels) buffer.putShort(samplesPerChannel[channel][frame])
        }
        val format = AudioFormat(sampleRate, 16, channels, true, false)
        AudioSystem.write(
            AudioInputStream(buffer.array().inputStream(), format, frames.toLong()),
            AudioFileFormat.Type.WAVE,
            path.toFile(),
        )
    }

    private fun tempWav(name: String): Path =
        Files.createTempDirectory("wav-reader-test").resolve(name)

    @Test
    fun `reads 16 kHz mono into normalized floats`() {
        val path = tempWav("mono.wav")
        writeWav(path, listOf(listOf(0, 16384, -16384, 32767)))

        val samples = WavReader.read(path)

        assertEquals(4, samples.size)
        assertTrue(abs(0f - samples[0]) < 1e-3f)
        assertTrue(abs(0.5f - samples[1]) < 1e-3f, "got ${samples[1]}")
        assertTrue(abs(-0.5f - samples[2]) < 1e-3f, "got ${samples[2]}")
        assertTrue(samples[3] in 0.99f..1.0f, "got ${samples[3]}")
    }

    @Test
    fun `keeps every sample in the normalized range`() {
        val path = tempWav("extremes.wav")
        writeWav(path, listOf(listOf(-32768, 32767, 0)))

        WavReader.read(path).forEach {
            assertTrue(it in -1.0f..1.0f, "sample $it escaped the normalized range")
        }
    }

    @Test
    fun `downmixes stereo to mono by averaging the channels`() {
        val path = tempWav("stereo.wav")
        writeWav(path, listOf(listOf(16384, 0), listOf(0, 16384)))

        val samples = WavReader.read(path)

        assertEquals(2, samples.size)
        assertTrue(abs(0.25f - samples[0]) < 1e-3f, "got ${samples[0]}")
        assertTrue(abs(0.25f - samples[1]) < 1e-3f, "got ${samples[1]}")
    }

    @Test
    fun `rejects a wrong sample rate instead of resampling`() {
        val path = tempWav("44k.wav")
        writeWav(path, listOf(listOf(0, 1000)), sampleRate = 44_100f)

        val failure = assertFailsWith<UnsupportedAudioException> { WavReader.read(path) }

        assertTrue(failure.message!!.contains("44100"), "message should name the actual rate")
        assertTrue(failure.message!!.contains("ffmpeg"), "message should give the fix command")
        assertTrue(failure.message!!.contains("44k.wav"), "message should name the file")
    }

    @Test
    fun `rejects a missing file with a clear message`() {
        val failure = assertFailsWith<UnsupportedAudioException> {
            WavReader.read(Path.of("does-not-exist.wav"))
        }

        assertTrue(failure.message!!.contains("does-not-exist.wav"))
    }

    @Test
    fun `reads a file with many samples`() {
        val path = tempWav("long.wav")
        writeWav(path, listOf(List(16_000) { (it % 1000).toShort() }))

        assertEquals(16_000, WavReader.read(path).size)
    }

    @Test
    fun `rejects a bit depth other than 16, naming the fix`() {
        val path = tempWav("8bit.wav")
        val samples = byteArrayOf(0, 64, -64)
        val format = AudioFormat(16_000f, 8, 1, true, false)
        AudioSystem.write(
            AudioInputStream(samples.inputStream(), format, samples.size.toLong()),
            AudioFileFormat.Type.WAVE,
            path.toFile(),
        )

        val failure = assertFailsWith<UnsupportedAudioException> { WavReader.read(path) }

        assertTrue(failure.message!!.contains("8-bit"), failure.message!!)
        assertTrue(failure.message!!.contains("ffmpeg"), "should give the fix command")
    }

    @Test
    fun `reports an unreadable file as unsupported rather than leaking a checked exception`() {
        val path = tempWav("not-audio.wav")
        path.writeText("this is not a wav file at all")

        val failure = assertFailsWith<UnsupportedAudioException> { WavReader.read(path) }

        assertTrue(failure.message!!.contains("not-audio.wav"))
    }
}
