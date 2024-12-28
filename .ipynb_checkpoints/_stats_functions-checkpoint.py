import io
import scipy.stats as stats
from itertools import combinations

def generate_dimension_combinations(dimensions):
    """
    Generate and print all combinations of two, three, and four dimensions.

    Args:
        dimensions (list): List of dimension names (strings).
    """
    if not dimensions or len(dimensions) < 2:
        print("At least two dimensions are required.")
        return

    print("\nCombinations of 2 dimensions:")
    for combo in combinations(dimensions, 2):
        print(combo)

    if len(dimensions) >= 3:
        print("\nCombinations of 3 dimensions:")
        for combo in combinations(dimensions, 3):
            print(combo)

    if len(dimensions) >= 4:
        print("\nCombinations of 4 dimensions:")
        for combo in combinations(dimensions, 4):
            print(combo)




def chi_squared_test(observed_male, total_male, observed_female, total_female):
    """
    Perform a chi-squared test for independence between gender and obesity.
    
    Parameters:
        observed_male (int): Number of obese males.
        total_male (int): Total number of males.
        observed_female (int): Number of obese females.
        total_female (int): Total number of females.

    Returns:
        float: The chi-squared statistic.
        float: The p-value of the test.
    """
    # Create observed and expected counts for contingency table
    observed_not_male = total_male - observed_male
    observed_not_female = total_female - observed_female

    observed = [
        [observed_male, observed_not_male],
        [observed_female, observed_not_female]
    ]

    # Perform the chi-squared test
    chi2_stat, p_value, _, _ = stats.chi2_contingency(observed)
    return chi2_stat, p_value


def absolute_percentage_difference(value1, value2):
    """
    Calculate the absolute percentage difference between two values.

    Parameters:
        value1 (float or int): The first value.
        value2 (float or int): The second value.

    Returns:
        float: The absolute percentage difference between the two values.
    """
    try:
        # Calculate the absolute difference
        absolute_difference = abs(value1 - value2)
        
        # Calculate the average of the two values
        average_value = (value1 + value2) / 2
        
        # Calculate and return the absolute percentage difference
        return (absolute_difference / average_value) * 100
    except ZeroDivisionError:
        return float('inf')  # Handle division by zero if both values are 0
        

def dimension_distribution(df, dimensions):
    """
    Generate distribution plots for categorical dimensions.

    Args:
        df (pd.DataFrame): Input DataFrame.
        dimensions (list): List of dimension names to analyze.
    """
    for dimension in dimensions:
        if dimension in df.columns:
            buffer = io.StringIO()
            counts = df[dimension].value_counts()
            percentages = df[dimension].value_counts(normalize=True) * 100

            buffer.write(f"Distribution for '{dimension}':\n")
            for category, count in counts.items():
                percentage = percentages[category]
                buffer.write(f"{category}: {round(percentage, 2)}% ({count})\n")

            print(buffer.getvalue())
        else:
            print(f"Dimension '{dimension}' not found in DataFrame.")

def two_dimension_subgroup_distributions(df, dimensions, total_count, tags=""):
    """
    Calculate and print subgroup distributions for two dimensions.

    Args:
        df (pd.DataFrame): Input DataFrame.
        dimensions (tuple): Tuple of two dimension names.
        total_count (int): Total count of dataset entries.
    """
    if len(dimensions) != 2:
        print("Please provide exactly two dimensions.")
        return

    dim1, dim2 = dimensions
    if dim1 not in df.columns or dim2 not in df.columns:
        print(f"One or both dimensions '{dim1}' and '{dim2}' not found in DataFrame.")
        return

    buffer = io.StringIO()
    buffer.write(f"Subgroup distributions for '{dim1}' and '{dim2}'{tags}:\n\n")

    for dim1_value in df[dim1].unique():
        buffer.write(f"{dim1_value}:\n")
        subgroup = df[df[dim1] == dim1_value][dim2].value_counts()
        subgroup_percentages = df[df[dim1] == dim1_value][dim2].value_counts(normalize=True) * 100

        for dim2_value, count in subgroup.items():
            percentage = subgroup_percentages[dim2_value]
            total_percentage = (count / total_count) * 100
            buffer.write(
                f"  {dim2_value}: {round(percentage, 2)}% ({count}) ({round(total_percentage, 2)}% of all dataset entries)\n"
            )
        buffer.write("\n")

    print(buffer.getvalue())


def three_dimension_subgroup_distributions(df, dimensions, total_count, tags=""):
    """
    Calculate and print subgroup distributions for three dimensions.

    Args:
        df (pd.DataFrame): Input DataFrame.
        dimensions (tuple): Tuple of three dimension names.
        total_count (int): Total count of dataset entries.
    """
    if len(dimensions) != 3:
        print("Please provide exactly three dimensions.")
        return

    dim1, dim2, dim3 = dimensions
    if dim1 not in df.columns or dim2 not in df.columns or dim3 not in df.columns:
        print(f"One or more dimensions '{dim1}', '{dim2}', or '{dim3}' not found in DataFrame.")
        return

    buffer = io.StringIO()
    buffer.write(f"Subgroup distributions for '{dim1}', '{dim2}', and '{dim3}'{tags}:\n\n")

    for dim1_value in df[dim1].unique():
        buffer.write(f"{dim1_value}:\n")
        for dim2_value in df[df[dim1] == dim1_value][dim2].unique():
            buffer.write(f"  {dim2_value}:\n")
            subgroup = df[(df[dim1] == dim1_value) & (df[dim2] == dim2_value)][dim3].value_counts()
            subgroup_percentages = df[(df[dim1] == dim1_value) & (df[dim2] == dim2_value)][dim3].value_counts(normalize=True) * 100

            for dim3_value, count in subgroup.items():
                percentage = subgroup_percentages[dim3_value]
                total_percentage = (count / total_count) * 100
                buffer.write(
                    f"    {dim3_value}: {round(percentage, 2)}% ({count}) ({round(total_percentage, 2)}% of all dataset entries)\n"
                )
            buffer.write("\n")

    print(buffer.getvalue())


def print_categorical_modes(df):
    """
    Prints the mode(s) for each categorical column in the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        None
    """
    # Select categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if categorical_cols.empty:
        print("No categorical columns found in the DataFrame.")
        return

    # Iterate over categorical columns and print the modes
    for col in categorical_cols:
        modes = df[col].mode().tolist()
        print(f"Column: {col}")
        print(f"Mode(s): {modes}")
        print("-")