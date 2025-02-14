package de.buw;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

public class App {

    public static void main(String[] args) throws IOException {

        if (args.length < 2) {
            System.out.println("Usage: java -jar <jarfile> <command> <args>");
            System.out.println("Commands: ");
            System.out.println(
                    "  alloySpecAnalysis <SpecDir> <outputPath>: Analyze Alloy specifications in the given dir and write the results to the output file");
            System.out.println(
                    "  alloyModelAnalysis <specPath> <cmdIndex> <outputPath> : Run Alloy models in the given path with the given command and write the results to the output file");
            System.out.println(
                    "  analyzeHalstead <filePath> : Analyze Halstead metrics for the given file");
            System.out.println(
                    "  analyzeErrors <filePath> <outputPath> : Analyze errors in the given Alloy file and write the results to the output file");
            System.exit(0);
        }
        String check = args[0];
        if (check.equals("alloySpecAnalysis") && args.length == 3) {
            new App().alloySpecAnalysis(args);
        } else if (check.equals("alloyModelAnalysis") && args.length == 4) {
            new App().alloyModelAnalysis(args);
        } else if (check.equals("analyzeHalstead")) {
            int[] hal = HalsteadAnalyzer.analyzeHalstead(args[1]);
            System.out.println(Arrays.toString(hal));
        } else if (check.equals("analyzeErrors")) {
            if (args.length == 3) {
                new App().alloyErrorAnalysis(args);
            } else {
                System.out.println("Invalid command or arguments");
            }
        } else {
            System.out.println("Invalid command or arguments");
        }


    }

    public void alloyErrorAnalysis(String[] args) throws IOException {
        String path = args[1];
        System.out.println("Checking: " + path);
        String[] errors = AlloyErrorsCategories.analyzeErrors(path);
        String report = args[1] + "," + errors[0] + "," + errors[1] + "\n";
        // Write header if the file is empty
        if (Files.notExists(Paths.get(args[2])) || Files.size(Paths.get(args[2])) == 0) {
            try {
                Files.writeString(Paths.get(args[2]), "spec,errorCategory,errorLocation\n",
                        StandardOpenOption.CREATE);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            Files.writeString(Paths.get(args[2]), report, StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void alloyModelAnalysis(String[] args) {
        String path = args[1];
        String cmdString = args[2];

        System.out.println("Checking: " + path);

        ModelAnalyzer ma = new ModelAnalyzer();
        String report = args[1] + "," + args[2] + ","
                + ma.analyzeWithTimeout(path, Integer.parseInt(cmdString), 60, TimeUnit.SECONDS);
        try {
            Files.writeString(Paths.get(args[3]), report, StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void alloySpecAnalysis(String[] args) {
        Path startDir = Paths.get(args[1]);
        String output = args[2];
        List<Path> specList = List.of();
        try {
            specList =
                    Files.walk(startDir).filter(Files::isRegularFile).collect(Collectors.toList());
        } catch (Exception e) {
            e.printStackTrace();
        }

        SpecAnalyzer specAnalyzer = new SpecAnalyzer();

        specList.forEach(spec -> {
            System.out.println(spec.toString());
            if (!spec.toString().endsWith(".als")) {
                return;
            }
            try {
                SpecAnalyzerResults results = specAnalyzer.analyze(spec.toString());

                // Open the file in append mode
                try (Writer writer = Files.newBufferedWriter(Paths.get(output),
                        StandardOpenOption.CREATE, StandardOpenOption.APPEND);
                        CSVPrinter csvPrinter = new CSVPrinter(writer, CSVFormat.DEFAULT)) {

                    // Check if the file is newly created, and write the header if needed
                    if (Files.size(Paths.get(output)) == 0) {
                        csvPrinter.printRecord("spec", "parseable", "eloc", "comments", "halstead",
                                "operators", "operands");
                    }

                    // Append the record to the CSV
                    csvPrinter.printRecord(spec, results.getSyntaxCheck(), results.getLoc(),
                            results.getNumberOfComments(), getHalsteadAsList(results.getHalstead()),
                            results.getOperators(), results.getOperands());
                    csvPrinter.flush();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
    }

    public static List<String> getHalsteadAsList(int[] intArray) {
        return Arrays.stream(intArray).mapToObj(String::valueOf).collect(Collectors.toList());
    }
}
