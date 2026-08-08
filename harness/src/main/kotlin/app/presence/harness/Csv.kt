package app.presence.harness

/**
 * A deliberately strict CSV reader: no quoting, no embedded commas, no escapes.
 *
 * Spec §9.1's fields are all simple tokens, and hand-rolled quote handling is a classic source
 * of silent corruption. This reader refuses anything ambiguous and names the offending line, so
 * a spreadsheet export that sneaks in a comma fails loudly instead of skewing the results.
 */
object Csv {

    fun parse(text: String): List<Map<String, String>> =
        parseWithLineNumbers(text).map { it.second }

    /**
     * Same as [parse], but pairs each row with its 1-based line number in the source text, so a
     * caller can name the line a human would find in their editor.
     */
    fun parseWithLineNumbers(text: String): List<Pair<Int, Map<String, String>>> {
        val lines = text.lineSequence()
            .withIndex()
            .filter { (_, line) -> line.isNotBlank() }
            .toList()
        require(lines.isNotEmpty()) { "CSV is empty: expected a header row" }

        val headerLineNumber = lines.first().index + 1
        val header = splitRow(lines.first().value, headerLineNumber)

        // Duplicate names would make zip-to-map keep only the last value, losing a column with no
        // error at all — the exact silent mis-parse this reader exists to prevent.
        val duplicates = header.groupingBy { it }.eachCount().filterValues { it > 1 }.keys
        require(duplicates.isEmpty()) {
            "line $headerLineNumber: duplicate column name(s) ${duplicates.joinToString()} — " +
                "each column must be unique or data is silently lost"
        }

        return lines.drop(1).map { (index, line) ->
            val lineNumber = index + 1
            val fields = splitRow(line, lineNumber)
            require(fields.size == header.size) {
                "line $lineNumber has ${fields.size} fields but the header has ${header.size}: $line"
            }
            lineNumber to header.zip(fields).toMap()
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
