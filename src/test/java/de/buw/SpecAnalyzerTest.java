package de.buw;

import org.junit.jupiter.api.Test;
import java.util.HashSet;
import java.util.Set;
import static org.junit.jupiter.api.Assertions.assertEquals;



class SpecAnalyzerTest {

    @Test
    void testCollectAlloyOperators() {
        String formula = "a <=> b";
        Set<String> allOperators = new HashSet<>();
        int[] operatorCount = {0};

        SpecAnalyzer.collectAlloyOperators(formula, allOperators, operatorCount);

        // Check if "<=>" is counted as a single operator
        assertEquals(1, allOperators.size());
        assertEquals(1, operatorCount[0]);
        assertEquals(true, allOperators.contains("<=>"));
    }

    @Test
    void testCollectAlloyOperands() {
        String formula = "a <=> b";
        Set<String> allOperands = new HashSet<>();
        int[] operandCount = {0};

        SpecAnalyzer.collectAlloyOperands(formula, allOperands, operandCount);

        // Check if "<=>" is counted as a single operator
        assertEquals(2, allOperands.size());
        assertEquals(2, operandCount[0]);
        assertEquals(true, allOperands.contains("a"));
    }

    @Test
    void testCollectAlloyOperatorsSquareBrackets() {
        String formula = "run {one nA[nA[A][A->A]]}";
        Set<String> allOperators = new HashSet<>(); // run, one, [, ->
        int[] operatorCount = {0};

        SpecAnalyzer.collectAlloyOperators(formula, allOperators, operatorCount);

        assertEquals(4, allOperators.size());
        assertEquals(6, operatorCount[0]);
        assertEquals(true, allOperators.contains("["));
        assertEquals(false, allOperators.contains("]"));
    }

    @Test
    void testCollectAlloyOperandsSquareBrackets() {
        String formula = "run {one nA[nA[A][A->A]]}";
        Set<String> allOperands = new HashSet<>();  // nA, A,
        int[] operandCount = {0}; // 5

        SpecAnalyzer.collectAlloyOperands(formula, allOperands, operandCount);

        assertEquals(2, allOperands.size());
        assertEquals(5, operandCount[0]);
        assertEquals(true, allOperands.contains("nA"));
        assertEquals(false, allOperands.contains("}"));
        assertEquals(false, allOperands.contains("]"));
    }

    @Test
    void testCollectAlloyOperatorsWithPrime() {
        String formula = "a' => b";
        Set<String> allOperators = new HashSet<>();
        int[] operatorCount = {0};

        SpecAnalyzer.collectAlloyOperators(formula, allOperators, operatorCount);

        // Check if "<=>" is counted as a single operator
        assertEquals(2, allOperators.size());
        assertEquals(2, operatorCount[0]);
        assertEquals(true, allOperators.contains("'"));
    }

    @Test
    void testCollectAlloyOperandWithPrime() {
        String formula = "a' => b";
        Set<String> allOperands = new HashSet<>();
        int[] operandCount = {0};

        SpecAnalyzer.collectAlloyOperands(formula, allOperands, operandCount);

        assertEquals(2, allOperands.size());
        assertEquals(2, operandCount[0]);
        assertEquals(true, allOperands.contains("a"));
    }

    @Test
    void testOperatorInNameIdentifier() {
        String formula = "fun Finder";
        Set<String> allOperators = new HashSet<>();
        int[] operatorCount = {0};

        SpecAnalyzer.collectAlloyOperators(formula, allOperators, operatorCount);

        assertEquals(1, allOperators.size());
        assertEquals(1, operatorCount[0]);
        assertEquals(false, allOperators.contains("in"));
    }

    @Test
    void testCollectAlloyOperatorsPred1() {
        String formula = "pred Initial [s: State]  { no s.holds + s.waits }"; // pred, [, no, ., +
        Set<String> allOperators = new HashSet<>();
        int[] operatorCount = {0};

        SpecAnalyzer.collectAlloyOperators(formula, allOperators, operatorCount);

        assertEquals(5, allOperators.size());
        assertEquals(6, operatorCount[0]);
        assertEquals(true, allOperators.contains("["));
    }

    @Test
    void testCollectAlloyOperandsPred1() {
        String formula = "pred Initial [s: State]  { no s.holds + s.waits }";
        Set<String> allOperands = new HashSet<>();
        int[] operandCount = {0};

        SpecAnalyzer.collectAlloyOperands(formula, allOperands, operandCount);

        assertEquals(5, allOperands.size());
        assertEquals(7, operandCount[0]);
        assertEquals(true, allOperands.contains("Initial"));
        assertEquals(true, allOperands.contains("State"));
    }

    @Test
    void testCollectAlloyOperatorsPred2() {
        String formula =
                "pred Add [me: DB, adv: Query, r: Record, db: DB] {no me.attributes & adv.attributes}";
        Set<String> allOperators = new HashSet<>(); // pred, [, no, ., &
        int[] operatorCount = {0}; // 6

        SpecAnalyzer.collectAlloyOperators(formula, allOperators, operatorCount);

        assertEquals(5, allOperators.size());
        assertEquals(6, operatorCount[0]);
        assertEquals(true, allOperators.contains("&"));
        assertEquals(false, allOperators.contains("Query"));
        assertEquals(false, allOperators.contains("Add"));
    }

    @Test
    void testCollectAlloyOperandsPred2() {
        String formula =
                "pred Add [me: DB, adv: Query, r: Record, db: DB] {no me.attributes & adv.attributes}";
        Set<String> allOperands = new HashSet<>(); // Add, me, db, DB, adv, Query, r, Record, attributes
        int[] operandCount = {0}; //13

        SpecAnalyzer.collectAlloyOperands(formula, allOperands, operandCount);

        assertEquals(9, allOperands.size());
        assertEquals(13, operandCount[0]);
        assertEquals(true, allOperands.contains("me"));
        assertEquals(true, allOperands.contains("Query"));
        assertEquals(false, allOperands.contains("["));
    }


    @Test
    void testCollectAlloyOperatorsFun() {
        String formula = """
                fun ClosestPrecedingFinger(s: State) {
                    all n: s.active | let nd = n.s.data
                    }
                """; // fun, all, ., let, =

        Set<String> allOperators = new HashSet<>();
        int[] operatorCount = {0};

        SpecAnalyzer.collectAlloyOperators(formula, allOperators, operatorCount);
        assertEquals(5, allOperators.size());
        assertEquals(7, operatorCount[0]);
        assertEquals(true, allOperators.contains("="));
    }

    @Test
    void testCollectAlloyOperandsFun() {
        String formula = """
                fun ClosestPrecedingFinger(s: State) {
                    all n: s.active | let nd = n.s.data
                    }
                """;
        Set<String> allOperands = new HashSet<>();
        int[] operandCount = {0};

        SpecAnalyzer.collectAlloyOperands(formula, allOperands, operandCount);

        assertEquals(7, allOperands.size());
        assertEquals(10, operandCount[0]);
        assertEquals(true, allOperands.contains("s"));
        assertEquals(false, allOperands.contains("|"));
    }


    @Test
    void testLineCommentRemoval1() {
        String formula = """
                fun ClosestPrecedingFinger(s: State) {
                    // all n: s.active | let nd = n.s.data
                    }
                """;
        String result = SpecAnalyzer.removeComments(formula);
        assertEquals(false, result.contains("all n: s.active | let nd = n.s.data"));
    }

    @Test
    void testLineCommentRemoval2() {
        String formula = """
                fun ClosestPrecedingFinger(s: State) {
                    -- all n: s.active | let nd = n.s.data
                    }
                """;
        String result = SpecAnalyzer.removeComments(formula);
        assertEquals(false, result.contains("all n: s.active | let nd = n.s.data"));
    }

    @Test
    void testLineCommentRemoval3() {
        String formula = """
                fun ClosestPrecedingFinger(s: State) {
                    all n: s.active | let nd = n.s.data -- comment
                    }
                """;
        String result = SpecAnalyzer.removeComments(formula);
        assertEquals(false, result.contains("comment"));
    }

    @Test
    void testBlockCommentRemoval1() {
        String formula = """
                /*
                * This is a block comment
                */
                fun ClosestPrecedingFinger(s: State) {
                    all n: s.active | let nd = n.s.data
                    }
                """;
        String result = SpecAnalyzer.removeComments(formula);
        assertEquals(false, result.contains("This is a block comment"));
    }

    @Test
    void testBlockCommentRemoval2() {
        String formula = """
                fun ClosestPrecedingFinger(s: State) {
                    all n: s.active | let nd = n.s.data /* Also a block comment */
                    }
                """;
        String result = SpecAnalyzer.removeComments(formula);
        assertEquals(false, result.contains("Also a block comment"));
    }

    @Test
    void testCountComments1() {
        String formula = """
                fun ClosestPrecedingFinger(s: State) {
                    all n: s.active | let nd = n.s.data /* Also a block comment */
                    }
                """;
        int result = SpecAnalyzer.getAlloyComments(formula);
        assertEquals(1, result);
    }

    @Test
    void testCountComments2() {
        String formula = """
                /*
                * This is a block comment
                */
                fun ClosestPrecedingFinger(s: State) {
                    all n: s.active | let nd = n.s.data /* Also a block comment */
                    }
                """;
        int result = SpecAnalyzer.getAlloyComments(formula);
        assertEquals(2, result);
    }

    @Test
    void testCountComments3() {
        String formula = """
                /*
                * This is a block comment
                */
                fun ClosestPrecedingFinger(s: State) {
                    // Line comment
                    -- Another line comment
                    all n: s.active | let nd = n.s.data /* Also a block comment */
                    }
                """;
        int result = SpecAnalyzer.getAlloyComments(formula);
        assertEquals(4, result);
    }

    @Test
    void testCountELOC1() {
        String formula = """
                /*
                * This is a block comment
                */
                fun ClosestPrecedingFinger(s: State) {
                    // Line comment
                    -- Another line comment
                    all n: s.active | let nd = n.s.data /* Also a block comment */
                    }
                """;
        int result = SpecAnalyzer.getAlloyLOC(formula);
        assertEquals(3, result);
    }

    @Test
    void testCountELOC2() {
        String formula =
                """
                        -- fun ClosestPrecedingFinger(s: State) {// Line comment all n: s.active | let nd = n.s.data /* Also a block comment */}
                        """;
        int result = SpecAnalyzer.getAlloyLOC(formula);
        assertEquals(0, result);
    }
}
