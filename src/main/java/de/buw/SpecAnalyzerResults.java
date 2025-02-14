package de.buw;

import java.util.Set;

public class SpecAnalyzerResults {
    private Boolean syntaxCheck;
    private String resultMessage;
    private int numberOfComments;
    private int loc;
    private int[] halstead;
    private Set<String> operators;
    private Set<String> operands;

    public Boolean getSyntaxCheck() {
        return syntaxCheck;
    }

    public void setSyntaxCheck(Boolean syntaxCheck) {
        this.syntaxCheck = syntaxCheck;
    }

    public String getResultMessage() {
        return resultMessage;
    }

    public void setResultMessage(String resultMessage) {
        this.resultMessage = resultMessage;
    }

    public int getNumberOfComments() {
        return numberOfComments;
    }

    public void setNumberOfComments(int numberOfComments) {
        this.numberOfComments = numberOfComments;
    }

    public int getLoc() {
        return loc;
    }

    public void setLoc(int loc) {
        this.loc = loc;
    }

    public int[] getHalstead() {
        return halstead;
    }

    public void setHalstead(int[] halstead) {
        this.halstead = halstead;
    }

    public Set<String> getOperators() {
        return operators;
    }

    public void setOperators(Set<String> operators) {
        this.operators = operators;
    }

    public Set<String> getOperands() {
        return operands;
    }

    public void setOperands(Set<String> operands) {
        this.operands = operands;
    }
}
