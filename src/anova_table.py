def anova_for_categorical_vars(dataset, dep_var, cat_vars):
    results = [] # initialize a list to store the results
    for var in cat_vars:
        model = smf.ols(
            f'{dep_var} ~ C({var})',
            data = dataset
        ).fit()
        
        anova_table = anova_lm(model)
        
        # extract the F-statistic and p-value from the ANOVA table
        f_statistic = anova_table['F'].iloc[0]
        p_value = anova_table['PR(>F)'].iloc[0]

        # append the results to the list
        results.append([var, f_statistic, p_value])

    # create a DataFrame with the results
    anova_results_df = pd.DataFrame(
        results,
        columns = ['Variable', 'F-statistic', 'p-value']
    )

    # sort the DataFrame by F-statistic in decreasing order
    # ascending = false, because we need to have most strongly correlated vars in first row
    anova_results_df = anova_results_df.sort_values(by = 'F-statistic', ascending = False)

    return anova_results_df