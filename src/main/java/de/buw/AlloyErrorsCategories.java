package de.buw;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.regex.Pattern;
import edu.mit.csail.sdg.alloy4.A4Reporter;
import edu.mit.csail.sdg.parser.CompModule;
import edu.mit.csail.sdg.parser.CompUtil;
import edu.mit.csail.sdg.translator.A4Options;

public class AlloyErrorsCategories {
    public static void main(String[] args) {
        String[] errors = analyzeErrors("data/code/fmp/30300.als");
        System.out.println(errors[0] + " " + errors[1]);
    }

    public static String[] analyzeErrors(String path) {
        A4Reporter rep = new A4Reporter();
        String spec = "";
        String errorCategory = "";
        String errorLocation = "";
        try {
            spec = new String(Files.readAllBytes(Paths.get(path)));
        } catch (Exception e) {
            e.printStackTrace();
        }

        try {
            CompModule module = CompUtil.parseEverything_fromFile(rep, null, path);
        } catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            String stackTrace = sw.toString();
            // Get Error Category
            if (stackTrace.contains("Type error")) {
                errorCategory = "Type Error";
            } else if (stackTrace.contains("Syntax error")) {
                errorCategory = "Syntax Error";
            } else {
                errorCategory = "Other Error";
            }

            // Get error location
            String[] stackTraceLines = stackTrace.split("\n");
            String firstLine = stackTraceLines[0];
            int indexline = firstLine.indexOf("line") + 5;
            int indexcolumn = firstLine.indexOf("column");
            String lineStr = "";
            lineStr = firstLine.substring(indexline, indexcolumn - 1);
            int lineInt = Integer.parseInt(lineStr);

            String[] linesSpec = spec.split("\\R"); // Cross-platform line separator
            for (int j = Math.min(lineInt, linesSpec.length - 1); j >= 0; j--) {
                String line = linesSpec[j].trim();
                if (Pattern.compile("^\\s*sig\\b").matcher(line).find()) {
                    errorLocation = "sig";
                    break;
                } else if (Pattern.compile("^\\s*fact\\b").matcher(line).find()) {
                    errorLocation = "fact";
                    break;
                } else if (Pattern.compile("^\\s*pred\\b").matcher(line).find()) {
                    errorLocation = "pred";
                    break;
                } else if (Pattern.compile("^\\s*assert\\b").matcher(line).find()) {
                    errorLocation = "assert";
                    break;
                } else if (Pattern.compile("^\\s*fun\\b").matcher(line).find()) {
                    errorLocation = "fun";
                    break;
                } else if (Pattern.compile("^\\s*check\\b").matcher(line).find()) {
                    errorLocation = "check";
                    break;
                } else if (Pattern.compile("^\\s*run\\b").matcher(line).find()) {
                    errorLocation = "run";
                    break;
                }
            }
        }
        return new String[] {errorCategory, errorLocation};

    }

    public static A4Options getOptions() {
        A4Options opt = new A4Options();
        opt.solver = A4Options.SatSolver.SAT4J;
        return opt;
    }
}
