package de.buw;

import java.util.HashSet;
import java.util.Set;

public class HalsteadAnalyzer {
    public static int[] analyzeHalstead(String modelPath) {
        Set<String> uniqueOperators = new HashSet<>();
        Set<String> uniqueOperands = new HashSet<>();
        int[] totalOperatorsCount = {0};
        int[] totalOperandsCount = {0};

        int[] halstead = new int[] {uniqueOperators.size(), uniqueOperands.size(),
                totalOperatorsCount[0], totalOperandsCount[0]};

        return halstead;

    }

}
