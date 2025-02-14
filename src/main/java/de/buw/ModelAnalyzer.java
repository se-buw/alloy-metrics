package de.buw;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import edu.mit.csail.sdg.alloy4.A4Reporter;
import edu.mit.csail.sdg.alloy4.SafeList;
import edu.mit.csail.sdg.ast.Command;
import edu.mit.csail.sdg.ast.Sig;
import edu.mit.csail.sdg.parser.CompModule;
import edu.mit.csail.sdg.parser.CompUtil;
import edu.mit.csail.sdg.translator.A4Options;
import edu.mit.csail.sdg.translator.A4Solution;
import edu.mit.csail.sdg.translator.TranslateAlloyToKodkod;

public class ModelAnalyzer {

    public String analyze(String modelPath, int cmdIndex) {
        String res = "";
        A4Options options = getOptions();
        A4Reporter rep = new A4Reporter();
        try {
            CompModule module = CompUtil.parseEverything_fromFile(rep, null, modelPath);
            Command cmd = module.getAllCommands().get(cmdIndex);
            SafeList<Sig> sigs = module.getAllSigs();
            A4Solution ans =
                    TranslateAlloyToKodkod.execute_command(A4Reporter.NOP, sigs, cmd, options);
            if (ans.satisfiable()) {
                res = "SAT";
            } else {
                res = "UNSAT";
            }
        } catch (Exception e) {
            e.printStackTrace();
            res = "PARSEERROR";
        }
        return res + "\n";

    }

    public String analyzeWithTimeout(String modelPath, int cmdIndex, long timeout,
            TimeUnit timeUnit) {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<String> future = executor.submit(() -> analyze(modelPath, cmdIndex));
        try {
            return future.get(timeout, timeUnit);
        } catch (TimeoutException e) {
            future.cancel(true);
            return "TIMEOUT\n";
        } catch (Exception e) {
            e.printStackTrace();
            return "ERROR\n";
        } finally {
            executor.shutdownNow();
        }
    }

    public static A4Options getOptions() {
        A4Options opt = new A4Options();
        opt.solver = A4Options.SatSolver.SAT4J;
        return opt;
    }
}
