package de.buw;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.mit.csail.sdg.alloy4.A4Reporter;
import edu.mit.csail.sdg.alloy4.Err;
import edu.mit.csail.sdg.ast.Command;
import edu.mit.csail.sdg.ast.Module;
import edu.mit.csail.sdg.parser.CompUtil;
import edu.mit.csail.sdg.translator.A4Options;
import edu.mit.csail.sdg.translator.A4Solution;
import edu.mit.csail.sdg.translator.TranslateAlloyToKodkod;

public class SpecAnalyzer {
    private static final String[] ALLOY_OPERATORS = {"=>", "<=>", "++", "=<", "->", ">=", "||", "[",
            "<:", ":>", "<", ">", "&&", "=", "+", "-", "&", ".", "~", "*", "^", "!", "#", ";", "'",
            "one", "lone", "some", "abstract", "all", "iff", "but", "else", "extends", "set",
            "implies", "module", "open", "and", "disj", "for", "in", "no", "or", "as", "sum",
            "exactly", "let", "not", "enum", "var", "steps", "always", "historically", "eventually",
            "once", "after", "before", "until", "since", "releases", "triggered", "check", "fact",
            "sig", "fun", "pred", "assert", "run"};

    public SpecAnalyzerResults analyze(String path) throws IOException {
        SpecAnalyzerResults results = new SpecAnalyzerResults();
        String spec = "";
        try {
            spec = new String(Files.readAllBytes(Paths.get(path)));
        } catch (Exception e) {
            e.printStackTrace();
        }
        Set<String> uniqueOperators = new HashSet<>();
        Set<String> uniqueOperands = new HashSet<>();
        int[] totalOperatorsCount = {0};
        int[] totalOperandsCount = {0};
        // Set<String> specialOperators = new HashSet<>();
        // Map<String, Integer> nameOccurrences = new HashMap<>();

        collectAlloyOperators(spec, uniqueOperators, totalOperatorsCount);
        collectAlloyOperands(spec, uniqueOperands, totalOperandsCount);

        // NOTE: We are not collecting special operators i.e., (fun, pred, assert) names as
        // operators
        // collectSpecialOperatorNamesAndCountOccurrencesAlloy(spec, specialOperators,
        // nameOccurrences);

        // // adding special operators to the allOperators
        // for (Map.Entry<String, Integer> entry : nameOccurrences.entrySet()) {
        // uniqueOperators.add(entry.getKey());
        // // adding special operators to the totalOperatorsCount
        // totalOperatorsCount[0] = totalOperatorsCount[0] + (entry.getValue() - 1);

        // // deleting special operators from the totalOperandsCount
        // totalOperandsCount[0] = totalOperandsCount[0] - (entry.getValue() - 1);
        // }

        int[] halstead = new int[] {uniqueOperators.size(), uniqueOperands.size(),
                totalOperatorsCount[0], totalOperandsCount[0]};
        String content = new String(Files.readAllBytes(Paths.get(path)));
        results.setSyntaxCheck(checkAlloySyntax(path));
        // results.setResultMessage(runAlloy(path));
        results.setNumberOfComments(getAlloyComments(content));
        results.setLoc(getAlloyLOC(content));
        results.setHalstead(halstead);
        results.setOperators(uniqueOperators);
        results.setOperands(uniqueOperands);
        return results;
    }

    public static int getAlloyLOC(String spec) {
        int locCount = 0;
        String noComments = removeComments(spec);
        try (BufferedReader br = new BufferedReader(new StringReader(noComments))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) {
                    locCount++;
                }
            }
        } catch (IOException e) {
            System.out.println("Error reading spec: " + e.getMessage());
        }
        return locCount;
    }

    public static int getAlloyComments(String model) {
        int count = 0;

        // Pattern to match multi-line comments (/* ... */)
        Pattern multiLinePattern = Pattern.compile("(?s)/\\*.*?\\*/");
        Matcher multiLineMatcher = multiLinePattern.matcher(model);

        while (multiLineMatcher.find()) {
            count++;
        }

        // Pattern to match single-line comments (-- or //)
        Pattern singleLinePattern = Pattern.compile("(?m)(--.*$|//.*$)");
        Matcher singleLineMatcher = singleLinePattern.matcher(model);

        while (singleLineMatcher.find()) {
            count++;
        }

        return count;
    }

    public static void collectAlloyOperators(String formula, Set<String> allOperators,
            int[] operatorCount) {
        String formulaWithoutComments = removeComments(formula);
        String[] lines = formulaWithoutComments.split("\n");

        List<String> operators = Arrays.asList(ALLOY_OPERATORS).stream()
                .sorted((a, b) -> Integer.compare(b.length(), a.length())).map(Pattern::quote)
                .toList();

        // Combined pattern with word boundaries and lookarounds to match operators
        // "[" and "'" have special meaning in regex, so they are added separately
        String combinedPattern = "\\b(" + String.join("|", operators) + ")\\b|(?<=[^\\w])("
                + String.join("|", operators) + ")(?=[^\\w])|\\[|(')";
        Pattern pattern = Pattern.compile(combinedPattern);

        for (String line : lines) {
            Matcher matcher = pattern.matcher(line);
            while (matcher.find()) {
                String matchedOperator = matcher.group();
                allOperators.add(matchedOperator);
                ++operatorCount[0];
            }
        }
    }

    public static void collectAlloyOperands(String formula, Set<String> allOperands,
            int[] operandCount) {
        String formulaWithoutComments = removeComments(formula);
        // Include hyphens within operand names.
        Pattern pattern = Pattern.compile("[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*+(?:_[a-zA-Z0-9]+)*");

        // Set of words (Operators) to be excluded
        Set<String> excludedWords = new HashSet<>(Arrays.asList(ALLOY_OPERATORS));
        // Everything that is not an operator is not an operand e.g., ( , ]
        excludedWords.add("]");

        for (String line : formulaWithoutComments.split("\n")) {
            Matcher matcher = pattern.matcher(line);
            while (matcher.find()) {
                String operand = matcher.group();
                if (!excludedWords.contains(operand)) {
                    // Add to global unique operands set for tracking unique operands.
                    allOperands.add(operand);
                    // Increment total operands count for each unique operand found including
                    // duplicates.
                    ++operandCount[0];
                }
            }

        }
    }

    public static void collectSpecialOperatorNamesAndCountOccurrencesAlloy(String formula,
            Set<String> specialOperators, Map<String, Integer> nameOccurrences) {
        String formulaWithoutComments = removeComments(formula);
        String[] lines = formulaWithoutComments.split("\n");
        Pattern pattern = Pattern.compile("\\b(fun|pred|assert)\\s+(\\w+)");

        for (String line : lines) {
            Matcher matcher = pattern.matcher(line);
            while (matcher.find()) {
                String operator = matcher.group(1); //
                String name = matcher.group(2); // The name following the operator

                // Add the combined operator and name to the set of special operators
                specialOperators.add(operator + " " + name);

                // Initialize or increment the count for this name in the occurrences map
                nameOccurrences.put(name, nameOccurrences.getOrDefault(name, 0) + 1);
            }
        }

        // After collecting names, count their occurrences throughout the formula
        for (String name : nameOccurrences.keySet()) {
            Pattern namePattern = Pattern.compile("\\b" + Pattern.quote(name) + "\\b");
            for (String line : lines) {
                Matcher nameMatcher = namePattern.matcher(line);
                while (nameMatcher.find()) {
                    // Increment the count for each match found
                    nameOccurrences.put(name, nameOccurrences.get(name) + 1);
                }
            }

            // Since the name was already counted once when initially found, subtract one to
            // adjust
            nameOccurrences.put(name, nameOccurrences.get(name) - 1);
        }
    }

    public static Boolean checkAlloySyntax(String alloyFilePath) {
        try {
            // Parse the Alloy model file
            CompUtil.parseEverything_fromFile(A4Reporter.NOP, null, alloyFilePath);
            return true;
        } catch (Err e) {
            System.out.println("Syntax error in Alloy model: " + e.getMessage());
            return false;
        }
    }

    public String runAlloy(String alloyFilePath) {

        String syntaxResult = new String();

        try {
            String[] args = new String[] {alloyFilePath};

            A4Reporter rep = new A4Reporter() {};

            for (String filename : args) {

                Module world = CompUtil.parseEverything_fromFile(rep, null, filename);

                A4Options options = new A4Options();

                options.solver = A4Options.SatSolver.SAT4J;

                for (Command command : world.getAllCommands()) {

                    // Execute the command
                    A4Solution ans = TranslateAlloyToKodkod.execute_command(rep,
                            world.getAllReachableSigs(), command, options);

                    if (ans.satisfiable()) {
                        syntaxResult =
                                "Instance found. Predicate is consistent. Use Alloy Analyzer to visualize the instances.";
                    }
                    if (!ans.satisfiable()) {
                        syntaxResult =
                                "Unsatisfiable. No instance found. Predicate may be inconsistent.";
                    }
                }
            }

        } catch (Err e) {
            // This block catches syntax errors or other issues in the Alloy model.
            syntaxResult = "runCode failed!";
        } catch (Exception e) {
        }

        return syntaxResult;

    }

    public static String removeComments(String model) {
        // Remove multi-line comments (/* ... */)
        model = model.replaceAll("(?s)/\\*.*?\\*/", "");

        // Remove single-line comments (-- and //) after code or at the start of a line
        model = model.replaceAll("(?m)(--.*$|//.*$)", "");

        return model;
    }
}
