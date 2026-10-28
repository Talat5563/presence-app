package app.presence.harness

import app.presence.domain.Language
import java.time.LocalDate
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class ChatTest {

    private val bullet = "\u0015"

    private val transcript = """
        |@UTF8
        |@Begin
        |@Languages:	eng
        |@Participants:	CHI Ross Target_Child, MAR Mark Brother, FAT Father
        |@Date:	15-JUN-1980
        |@Media:	010106a, audio
        |*CHI:	more cookie . ${bullet}1000_2500$bullet
        |%mor:	qn|more n|cookie .
        |*FAT:	you want
        |	another one ? ${bullet}2600_4000$bullet
        |*MAR:	me too .
        |*CHI:	backwards . ${bullet}5000_5000$bullet
        |@End
    """.trimMargin()

    @Test
    fun `reads headers`() {
        val parsed = Chat.parse(transcript)

        assertEquals("010106a", parsed.mediaName)
        assertEquals(Language.ENGLISH, parsed.primaryLanguage)
        assertEquals(LocalDate.of(1980, 6, 15), parsed.date)
    }

    @Test
    fun `reads participants with and without a name`() {
        val parsed = Chat.parse(transcript)

        assertEquals(ChatParticipant("CHI", "Ross", "Target_Child"), parsed.participants["CHI"])
        assertEquals(ChatParticipant("FAT", null, "Father"), parsed.participants["FAT"])
    }

    @Test
    fun `reads time bullets, including on a continuation line`() {
        val parsed = Chat.parse(transcript)

        assertEquals(
            listOf(ChatUtterance("CHI", 1_000, 2_500), ChatUtterance("FAT", 2_600, 4_000)),
            parsed.utterances,
        )
    }

    @Test
    fun `counts unaligned and zero-length utterances rather than inventing times`() {
        assertEquals(2, Chat.parse(transcript).unalignedUtterances)
    }

    @Test
    fun `unlinked media cannot be cut`() {
        assertNull(Chat.parse("@Media:\tfoo, audio, unlinked\n").mediaName)
    }

    @Test
    fun `Hebrew is recognised and unsupported languages are not`() {
        assertEquals(Language.HEBREW, Chat.parse("@Languages:\theb\n").primaryLanguage)
        assertNull(Chat.parse("@Languages:\tfra\n").primaryLanguage)
    }

    @Test
    fun `tolerates a byte order mark and Windows line endings`() {
        val parsed = Chat.parse("﻿@Media:\tx, audio\r\n*CHI:\thi . ${bullet}0_900$bullet\r\n")

        assertEquals("x", parsed.mediaName)
        assertEquals(listOf(ChatUtterance("CHI", 0, 900)), parsed.utterances)
    }
}
