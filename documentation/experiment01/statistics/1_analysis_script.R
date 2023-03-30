""" 
This R code carries out statistical analyses on behalf of experiment01 data from folder './0_raw'.
For this, it loads relevant material and key-performance-indicators collected by experiment01.
Then, statistical analyses are carried out and stored as csv file '2_analysis_results.csv'.

    Copyright (C) 2022>  Dr.-Ing. Marcus Grum

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
"""

# __author__ = 'Marcus Grum, marcus.grum@uni-potsdam.de'
# SPDX-License-Identifier: AGPL-3.0-or-later or individual license
# SPDX-FileCopyrightText: 2022 Marcus Grum <marcus.grum@uni-potsdam.de>

# load relevant libraries
library(dplyr)
library(tidyr)
library(readr)
library(readxl)
library(psych)
library(effsize)
library(plyr)
library (car)

# set working directory to script directory
if(rstudioapi::isAvailable()){
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))  
}

# specify location of relevant data
###################################

# - experiment01
experiment01_filePath <- "./0_raw/All_Experiments_statistical_raw_data.csv"

# import relevant data
######################

# import experiment data
raw_data_experiment01 <- read_delim(experiment01_filePath, "\t", escape_double = FALSE, trim_ws = TRUE)

# specifying function for group analysis
custom_group_analysis <- function(id, current_OutputVariable, current_LearningPhase, current_InputFunctionGroup1, current_InputFunctionGroup2, current_LearningIteration, verbose) {
  
  verbose = TRUE
  decimalPlaces = 3
  
  # filtering relevant cases
  current_LearningPhaseAndIteration_data_experiment01 <- raw_data_experiment01 %>% filter(Input_LearningPhase==current_LearningPhase) %>% filter(Input_LearningIteration==current_LearningIteration)
  current_LearningPhaseAndIteration_data_experiment01_reducedBaseline <- current_LearningPhaseAndIteration_data_experiment01[which(current_LearningPhaseAndIteration_data_experiment01$Input_Function == current_InputFunctionGroup1 | current_LearningPhaseAndIteration_data_experiment01$Input_Function == current_InputFunctionGroup2),] %>% select(Input_Function, current_OutputVariable)
  
  # focus on basic summary statistics by groups
  current_descriptive_values = describeBy(as.numeric(unlist(current_LearningPhaseAndIteration_data_experiment01_reducedBaseline[current_OutputVariable])), current_LearningPhaseAndIteration_data_experiment01_reducedBaseline$Input_Function)
  #if (verbose) print(current_descriptive_values)
  current_group1_name = current_InputFunctionGroup1
  current_group1_mean = current_descriptive_values[[current_InputFunctionGroup1]]$mean
  current_group1_sd   = current_descriptive_values[[current_InputFunctionGroup1]]$sd
  current_group1_n    = current_descriptive_values[[current_InputFunctionGroup1]]$n
  if (verbose) print(paste0("  current_group1_name = ", current_group1_name, 
                            ", current_group1_mean = ", current_group1_mean, 
                            ", current_group1_sd = ", current_group1_sd, 
                            ", current_group1_n = ", current_group1_n))
  current_group2_name = current_InputFunctionGroup2
  current_group2_mean = current_descriptive_values[[current_InputFunctionGroup2]]$mean
  current_group2_sd   = current_descriptive_values[[current_InputFunctionGroup2]]$sd
  current_group2_n    = current_descriptive_values[[current_InputFunctionGroup2]]$n
  if (verbose) print(paste0("  current_group2_name = ", current_group2_name, 
                            ", current_group2_mean = ", current_group2_mean, 
                            ", current_group2_sd = ", current_group2_sd, 
                            ", current_group2_n = ", current_group2_n))
  
  # focus on variance homogeneity by Levene-test as t-test condition
  current_levene_result = leveneTest(y = current_LearningPhaseAndIteration_data_experiment01_reducedBaseline[[current_OutputVariable]], g=current_LearningPhaseAndIteration_data_experiment01_reducedBaseline$Input_Function)
  print(current_levene_result)
  current_levene_result_df = current_levene_result$Df[2]
  current_levene_result_F  = current_levene_result$`F value`[1]
  current_levene_result_p  = current_levene_result$`Pr(>F)`[1]
  if(is.nan(current_levene_result_p)) {
    current_levene_result_significance = FALSE
    current_levene_result_interpretation = 't-test'
  } else {
    current_levene_result_significance = if (abs(current_levene_result_p) < 0.05) TRUE else FALSE
    current_levene_result_interpretation = if (abs(current_levene_result_p) < 0.05) 'Welch' else 't-test'
  }
  current_levene_result_varianceEquality = if (current_levene_result_significance) FALSE else TRUE
  if (verbose) print(paste0("current_levene_result_df = ", current_levene_result_df, 
                            ", current_levene_result_F = ", current_levene_result_F,
                            ", current_levene_result_p = ", current_levene_result_p,
                            ", current_levene_result_significance = ", current_levene_result_significance
                            ))
  # focus on two-sided t-test
  t_test_result = t.test(current_LearningPhaseAndIteration_data_experiment01_reducedBaseline[[current_OutputVariable]] ~ current_LearningPhaseAndIteration_data_experiment01_reducedBaseline$Input_Function, var.equal = current_levene_result_varianceEquality, alternative = "two.sided", paired = FALSE)
  #if (verbose) print(t_test_result)
  current_t_value_emp = t_test_result$statistic
  current_degreeOfFreedom = t_test_result$parameter
  current_t_value_krit = 3.3749 # identify relevant t-values by https://www.easycalculation.com/statistics/t-distribution-critical-value-table.php
  if (current_degreeOfFreedom == 118) current_t_value_krit = 3.3749
  if (current_degreeOfFreedom ==  59) current_t_value_krit = 3.4632
  if ((current_degreeOfFreedom >  59) && (current_degreeOfFreedom <  60)) current_t_value_krit = 3.4602
  if ((current_degreeOfFreedom >  92) && (current_degreeOfFreedom <  93)) current_t_value_krit = 3.3995
  
  
  if(is.nan(current_t_value_emp)) {
    current_t_emp_greater_than_t_krit = FALSE
  } else {
    current_t_emp_greater_than_t_krit = if (abs(current_t_value_emp) > current_t_value_krit) TRUE else FALSE
  }
  current_H0_acceptance = if (current_t_emp_greater_than_t_krit) "no" else "yes"
  current_intervention_working = if (current_H0_acceptance != "yes") "accomplished" else "failed"
  current_p_value = t_test_result$p.value
  current_estimate = t_test_result$estimate
  if (verbose) print(paste0("  current_t_value_emp = ", current_t_value_emp, 
                            ", current_degreeOfFreedom = ", current_degreeOfFreedom,
                            ", current_t_value_krit = ", current_t_value_krit,
                            ", current_t_emp_greater_than_t_krit = ", current_t_emp_greater_than_t_krit,
                            ", current_H0_acceptance = ", current_H0_acceptance,
                            ", current_intervention_working = ", current_intervention_working,
                            ", current_p_value = ", current_p_value, 
                            ", current_estimate[1] = ", current_estimate[1], 
                            ", current_estimate[2] = ", current_estimate[2]))
  
  # focus on Cohen's d and Hedges g effect size
  ## compute Cohen's d
  current_effectSizeCohen = cohen.d(as.numeric(unlist(current_LearningPhaseAndIteration_data_experiment01_reducedBaseline[current_OutputVariable])), current_LearningPhaseAndIteration_data_experiment01_reducedBaseline$Input_Function)
  #if (verbose) print(current_effectSizeCohen)
  current_d_value = current_effectSizeCohen$estimate
  current_d_size  = current_effectSizeCohen$magnitude
  if (verbose) print(paste0("  current_d_value = ", current_d_value, 
                            ", current_d_size = ", current_d_size))
  ## compute Hedges' g
  current_effectSizeHedge = cohen.d(as.numeric(unlist(current_LearningPhaseAndIteration_data_experiment01_reducedBaseline[current_OutputVariable])), current_LearningPhaseAndIteration_data_experiment01_reducedBaseline$Input_Function, hedges.correction=TRUE)
  #if (verbose) print(current_effectSizeHedge)
  current_g_value = current_effectSizeHedge$estimate
  current_g_size  = current_effectSizeHedge$magnitude
  if (verbose) print(paste0("  current_g_value = ", current_g_value, 
                            ", current_g_size = ", current_g_size))
  
  returnList = list(
    current_group1_name, format(round(current_group1_mean, digits = decimalPlaces), nsmall = decimalPlaces), format(round(current_group1_sd, digits = decimalPlaces), nsmall = decimalPlaces), current_group1_n,
    current_group2_name, format(round(current_group2_mean, digits = decimalPlaces), nsmall = decimalPlaces), format(round(current_group2_sd, digits = decimalPlaces), nsmall = decimalPlaces), current_group2_n,
    format(round(current_t_value_emp, digits = decimalPlaces), nsmall = decimalPlaces), format(round(as.numeric(current_degreeOfFreedom), digits = decimalPlaces), nsmall = decimalPlaces), current_t_value_krit, current_t_emp_greater_than_t_krit, current_H0_acceptance, current_intervention_working, current_p_value,
    format(round(current_d_value, digits = decimalPlaces), nsmall = decimalPlaces),  as.character(current_d_size), format(round(current_g_value, digits = decimalPlaces), nsmall = decimalPlaces), as.character(current_g_size),
    current_levene_result_df, format(round(current_levene_result_F, digits = decimalPlaces), nsmall = decimalPlaces), format(round(current_levene_result_p, digits = decimalPlaces), nsmall = decimalPlaces), current_levene_result_significance, current_levene_result_interpretation
  )

  return(returnList)
}

verbose = FALSE

# initializing variable stack
OutputVariables       = c("Output_Accuracy", "Output_Loss")   # "Output_Accuracy", "Output_Loss"
LearningPhases        = c("training", "testing")              # "training", "testing"
InputFunctionGroups1  = c("Bias", "Manipulation", "Baseline") # "Bias", "Manipulation", "Baseline"
InputFunctionGroups2  = c("Bias", "Manipulation", "Baseline") # "Bias", "Manipulation", "Baseline"
LearningIterations    = c(0,5,11)                             # 0...11

# looping to all group combinations for realizing custom analysis
id <- 0
for (current_OutputVariable in OutputVariables) {
  for (current_LearningPhase in LearningPhases){
    for (current_InputFunctionGroup1 in InputFunctionGroups1){
      for (current_InputFunctionGroup2 in InputFunctionGroups2){
        for (current_LearningIteration in LearningIterations){
          if(current_InputFunctionGroup1 == current_InputFunctionGroup2){
            # remove self-entries
          }
          else if( (current_InputFunctionGroup1 == "Manipulation") && (current_InputFunctionGroup2 == "Bias") ){
            # remove redundant entries
          }
          else if( (current_InputFunctionGroup1 == "Baseline") && (current_InputFunctionGroup2 == "Bias") ){
            # remove redundant entries
          }
          else if( (current_InputFunctionGroup1 == "Baseline") && (current_InputFunctionGroup2 == "Manipulation") ){ 
            # remove redundant entries
          }
          else {
            print(paste0("realizing group comparison ", id, ": ", current_OutputVariable, " ", current_LearningPhase, " ", current_InputFunctionGroup1, " ", current_InputFunctionGroup2, " ", current_LearningIteration, "..."))
            current_result = custom_group_analysis(id, current_OutputVariable, current_LearningPhase, current_InputFunctionGroup1, current_InputFunctionGroup2, current_LearningIteration, verbose)
            #print(paste0("   result=", current_result))
            
            # pack and write csv data
            current_metaData = list(id, current_OutputVariable, current_LearningPhase, current_LearningIteration)
            current_result <- c(current_metaData, current_result)
            if (id == 0) {
              current_column_titles = list(
                "id", "OutputVariable", "LearningPhase", "LearningIteration",
                "group1_name", "group1_mean", "group1_sd", "group1_n",
                "group2_name", "group2_mean", "group2_sd", "group2_n",
                "t_value_emp", "degreeOfFreedom", "t_value_krit", "t_emp_greater_than_t_krit", "H0_acceptance", "intervention_working", "p_value",
                "d_value",  "d_size", "g_value", "g_size",
                "levene_df", "levene_F", "levene_p", "levene_signif", "current_levene_result_interpretation"
              )
              write.table(as.data.frame(current_column_titles), file="2_analysis_results.csv", quote=F, sep=",", row.names=FALSE, col.names=FALSE)
            }
            write.table(as.data.frame(current_result), file="2_analysis_results.csv", append=TRUE, quote=F, sep=",", row.names=FALSE, col.names=FALSE)
            
            # increment unique identifier
            id = id + 1
          }
        }
      }
    }
  }
}
