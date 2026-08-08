package app.presence.harness

/**
 * A deliberately strict CSV reader: no quoting, no embedded commas, no escapes.
 *
 * Spec §9.1's fields are all simple tokens, and hand-rolled quote handling is a classic source
 * of silent corruption. This reader refuses anything ambiguous and names the offending line, so
 * a spreadsheet export that sneaks in a comma fails loudly instead of skewing the results.
 */
object Csv {

    fun parse(text: String): List<Map<String, String>> {
        val lines = text.lineSequence()
            .withIndex()
            .filter { (_, line) -> line.isNotBlank() }
            .toList()
        require(lines.isNotEmpty()) { "CSV is empty: expected a header row" }

        val header = splitRow(lines.first().value, lineNumber = lines.first().index + 1)

        return lines.drop(1).map { (index, line) ->
            val lineNumber = index + 1
            val fields = splitRow(line, lineNumber)
            require(fields.size == header.size) {
                "line $lineNumber has ${fields.size} fields but the header has ${header.size}: $line"
            }
            header.zip(fields).toMap()
        }
    }

    private fun splitRow(line: String, lineNumber: Int): List<String> {
        require(!line.contains('"')) {
            "line $lineNumber contains a quote character; quoting is unsupported because fields " +
                "must not contain commas: $line"
        }
        return line.split(',').map { it.trim() }
    }
}
