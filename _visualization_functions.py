import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from _args import parse_args

args = parse_args()

def histogram_numerical_df_quartiles(df, tags=""):
    sns.set_theme(style=args.style, palette=args.palette, context=args.context) # , font=args.font

    for column in df.columns:
        # Check if the column is numeric
        if np.issubdtype(df[column].dtype, np.number):
            fig, ax = plt.subplots(figsize=(8, 6))

            # Calculate quartiles for binning
            quartiles = np.percentile(df[column].dropna(), [0, 25, 50, 75, 100])

            # Plot the histogram with custom bins
            counts, bins_edges, patches = ax.hist(df[column].dropna(), bins=quartiles, edgecolor="black")

            # Assign colors to each bin from the palette
            num_bins = len(patches)  # Get the actual number of bins used
            colors = sns.color_palette(n_colors=num_bins)
            for color, patch in zip(colors, patches):
                patch.set_facecolor(color)

            # Add count labels and min-max values on each bar
            for count, bin_edge_start, bin_edge_end in zip(counts, bins_edges[:-1], bins_edges[1:]):
                # Calculate the midpoint of the bin for positioning text
                bin_mid = (bin_edge_start + bin_edge_end) / 2
                
                # Add count label above the bar
                ax.annotate(f'{int(count)}',
                            xy=(bin_mid, count), 
                            ha='center', va='bottom', fontsize=10)

                # Add min-max values below the bar
                ax.annotate(f'[{bin_edge_start:.0f} - {bin_edge_end:.0f}]',
                            xy=(bin_mid, count * 0.1), 
                            ha='center', va='top', fontsize=11, color='black')

            # Add padding and format title
            fig.suptitle(f"'{column}' by quartiles {tags}", fontsize=16, weight='bold', y=0.95)
            fig.canvas.manager.set_window_title(f'Histogram for {column}')

            # Add labels and grid
            ax.set_xlabel(column, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.show()
        else:
            print(f"Column '{column}' is not numerical and will be skipped.")

def histogram_numerical_df(df, bins="auto", tags=""):
    sns.set_theme(style=args.style, palette=args.palette, context=args.context) # , font=args.font

    for column in df.columns:
        # Check if the column is numeric
        if np.issubdtype(df[column].dtype, np.number):
            fig, ax = plt.subplots(figsize=(6, 6))

            # Define bins based on parameter (e.g., "auto", integer count)
            counts, bins_edges, patches = ax.hist(df[column].dropna(), bins=bins, edgecolor="black")

            # Assign colors to each bin from the palette
            num_bins = len(patches)  # Get the actual number of bins used
            colors = sns.color_palette(n_colors=num_bins)
            for color, patch in zip(colors, patches):
                patch.set_facecolor(color)

            # Add count labels on each bar
            for count, bin_edge in zip(counts, bins_edges[:-1]):
                ax.annotate(f'{int(count)}',
                            xy=(bin_edge + (bins_edges[1] - bins_edges[0]) / 2, count), 
                            ha='center', va='bottom', fontsize=10)
                
            # Add x and y labels
            ax.set_xlabel(column, fontsize=12, weight='bold')
            ax.set_ylabel('Frequency', fontsize=12, weight='bold')
            
            # Add padding and format title
            fig.suptitle(f"Histogram for '{column}' {tags}", fontsize=14, weight='bold', y=0.95)
            fig.canvas.manager.set_window_title(f'Histogram for {column}')
            
            plt.show()
        else:
            print(f"Column '{column}' is not numerical and will be skipped.")

def pie_chart_categorical_df(df, tags=""):
    sns.set_theme(style=args.style, palette=args.palette, context=args.context)

    for column in df.columns:
        unique_values = df[column].dropna().unique()
        
        if df[column].dtype == 'object' or len(unique_values) <= 10:  # Treat as categorical if non-numeric or few unique values
            counts = df[column].value_counts()
            labels = counts.index
            sizes = counts.values

            fig, ax = plt.subplots(figsize=(6, 6))

            # Assign specific colors to unique values
            palette = sns.color_palette(args.palette, len(unique_values))
            color_mapping = dict(zip(unique_values, palette))
            colors = [color_mapping[label] for label in labels]

            # Create labels with counts
            labels_with_counts = [f"{label} ({count})" for label, count in zip(labels, sizes)]

            # Create the pie chart
            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels_with_counts,
                autopct='%1.0f%%',
                startangle=90,
                colors=colors,
                textprops={'fontsize': 12}
            )

            # Customize the title
            ax.set_title(f"Distribution of '{column}' {tags}", fontsize=14, weight='bold', y=0.95)

            plt.show()

def histogram_categorical_df(df, rotation=0, tags=""):
    sns.set_theme(style=args.style, palette=args.palette, context=args.context) # , font=args.font

    for column in df.columns:
        unique_values = df[column].dropna().unique()

        
        # Rank bars by count
        counts = df[column].value_counts()
        sorted_categories = counts.index.tolist()
        df[column] = pd.Categorical(df[column], categories=sorted_categories, ordered=True)


        fig, ax = plt.subplots(figsize=(6, 6))

        if df[column].dtype == 'object' or len(unique_values) <= 10:  # Treat as categorical if non-numeric or few unique values
            # Use bar chart for categorical data
            total_count = len(df[column].dropna())
            # sns.countplot(x=column, data=df, ax=ax)
            
            # Assign specific colors to unique values
            palette = sns.color_palette(args.palette, len(unique_values))
            color_mapping = dict(zip(unique_values, palette))
            
            sns.countplot(x=column, hue=column, data=df, ax=ax, palette=color_mapping, legend=False)

            ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='center', fontsize=12) # , ha='right'

            # Add count labels on each bar
            for p in ax.patches:
                count = int(p.get_height())
                ax.annotate(f'{count}', 
                            (p.get_x() + p.get_width() / 2, p.get_height()), 
                            ha='center', va='bottom', fontsize=10)

            # Add percentage labels on each bar
            for p in ax.patches:
                count = int(p.get_height())
                if count > 0:  # Only annotate non-zero bars
                    percentage = count / total_count * 100
                    bin_mid = p.get_x() + p.get_width() / 2
    
                    # Add percentage label below the bar
                    ax.annotate(f'{percentage:.0f}%', 
                                xy=(bin_mid, p.get_height() * 0.001),  #0.1
                                ha='center', va='top', fontsize=10, color='black')

            fig.suptitle(f"Distribution of '{column} {tags}'", fontsize=14, weight='bold', y=0.95)
        plt.show()
        '''
        else:
            # Use histogram for continuous data
            num_bins = len(unique_values) if len(unique_values) < 30 else 10
            colors = sns.color_palette(n_colors=num_bins)

            counts, bins_edges, patches = ax.hist(df[column].dropna(), bins=num_bins, edgecolor="black")
            total_count = sum(counts)

            for color, patch in zip(colors, patches):
                patch.set_facecolor(color)

            # Add count and percentage labels on each bar
            for count, bin_edge_start, bin_edge_end in zip(counts, bins_edges[:-1], bins_edges[1:]):
                bin_mid = (bin_edge_start + bin_edge_end) / 2
                percentage = count / total_count * 100

                # Add count and percentage label below the bar
                ax.annotate(f'{percentage:.1f}%',
                            xy=(bin_mid, count * 0.1), 
                            ha='center', va='top', fontsize=11, color='black')

                # Add min-max values below the bar
                ax.annotate(f'[{bin_edge_start:.0f} - {bin_edge_end:.0f}]',
                            xy=(bin_mid, count * 0.05), 
                            ha='center', va='top', fontsize=11, color='black')

            fig.suptitle(f'Histogram for {column}', fontsize=14, weight='bold', y=0.95)
            '''
        # fig.canvas.manager.set_window_title(f'Chart for {column}')
        # plt.show()


def line_plots_grouped(df, target_dim, cat_dim, bins="auto", interpolation=False, density=False):
    """
    Create line plots of target dimension distributions grouped by categorical dimension.

    Parameters:
    - df: pandas.DataFrame
        Input dataframe containing the data.
    - target_dim: str
        The numerical column to analyze (e.g., age).
    - cat_dim: str
        The categorical column to group by (e.g., gender).
    - palette: str or list
        Color palette for the plot.
    - bins: str or int
        Binning strategy for grouping target dimension (e.g., "auto", integer count).
    """
    sns.set_theme(style=args.style, palette=args.palette, context=args.context) # , font=args.font

    if target_dim not in df.columns or cat_dim not in df.columns:
        raise ValueError("Target or categorical dimension not found in DataFrame.")

    if not np.issubdtype(df[target_dim].dtype, np.number):
        raise ValueError(f"The target dimension '{target_dim}' must be numerical.")

    if not pd.api.types.is_categorical_dtype(df[cat_dim]) and not pd.api.types.is_object_dtype(df[cat_dim]):
        raise ValueError(f"The categorical dimension '{cat_dim}' must be categorical or object type.")

    # Drop rows with missing values in the relevant columns
    df = df[[target_dim, cat_dim]].dropna()

    # Group data by the categorical column
    groups = df.groupby(cat_dim)

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 6))

    if interpolation:
        for category, group_data in groups:
            counts, bin_edges = np.histogram(group_data[target_dim], bins=bins, density=density)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            
            # Interpolate
            interp_func = interp1d(bin_centers, counts, kind='cubic')
            fine_bins = np.linspace(bin_centers.min(), bin_centers.max(), 500)
            smooth_counts = interp_func(fine_bins)
            
            # Plot
            ax.plot(fine_bins, smooth_counts, label=f'{cat_dim}: {category}')
    else:
        for category, group_data in groups:
            counts, bin_edges = np.histogram(group_data[target_dim], bins=bins, density=density)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax.plot(bin_centers, counts, label=f'{cat_dim}: {category}')

    ax.set_title(f"Distributions of '{target_dim}' grouped by '{cat_dim}'", fontsize=14, weight="bold")
    ax.set_xlabel(target_dim, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(title=cat_dim, fontsize=10)

    plt.tight_layout()
    plt.show()


def line_plots_kde(df, target_dim, cat_dim):
    """
    Create KDE line plots of target dimension distributions grouped by categorical dimension.

    Parameters:
    - df: pandas.DataFrame
        Input dataframe containing the data.
    - target_dim: str
        The numerical column to analyze (e.g., age).
    - cat_dim: str
        The categorical column to group by (e.g., gender).
    - palette: str or list
        Color palette for the plot.
    """
    sns.set_theme(style=args.style, palette=args.palette, context=args.context) # , font=args.font

    if target_dim not in df.columns or cat_dim not in df.columns:
        raise ValueError("Target or categorical dimension not found in DataFrame.")

    if not np.issubdtype(df[target_dim].dtype, np.number):
        raise ValueError(f"The target dimension '{target_dim}' must be numerical.")

    if not pd.api.types.is_categorical_dtype(df[cat_dim]) and not pd.api.types.is_object_dtype(df[cat_dim]):
        raise ValueError(f"The categorical dimension '{cat_dim}' must be categorical or object type.")

    # Drop rows with missing values in the relevant columns
    df = df[[target_dim, cat_dim]].dropna()

    # Group data by the categorical column
    groups = df.groupby(cat_dim)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    for category, group_data in groups:
        sns.kdeplot(data=group_data[target_dim], label=f'{cat_dim}: {category}', ax=ax, fill=False)

    ax.set_title(f"KDE of {target_dim} grouped by {cat_dim}", fontsize=14, weight="bold")
    ax.set_xlabel(target_dim, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(title=cat_dim, fontsize=10)

    plt.tight_layout()
    plt.show()


def stacked_binned_bar_pivot_table_reverse_axis(df, feature_1, feature_2, tag_1="1", tag_2="2", binning="quartiles", tags=""):
    sns.set_theme(style=args.style, palette=args.palette) #, font=args.font
    # Determine bin edges based on specified binning method
    if binning == "quartiles":
        bins = pd.qcut(df[feature_1], q=4, labels=["Q1", "Q2", "Q3", "Q4"], retbins=True)
        df['binned_feature_1'] = bins[0]
        bin_edges = bins[1]
    elif binning == "deciles":
        bins = pd.qcut(df[feature_1], q=10, labels=[f"D{i+1}" for i in range(10)], retbins=True)
        df['binned_feature_1'] = bins[0]
        bin_edges = bins[1]
    else:
        raise ValueError("Invalid binning method. Choose 'quartiles' or 'deciles'.")

    # Create a pivot table with counts and normalize to percentages
    pivot_table = df.pivot_table(index=feature_2, columns='binned_feature_1', aggfunc='size', fill_value=0)
    pivot_table_percent = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100  # Convert to percentages

    # Define colors using a Seaborn palette
    palette = sns.color_palette(args.palette, n_colors=len(pivot_table_percent.columns))
    colors = {category: palette[i] for i, category in enumerate(pivot_table_percent.columns)}

    # Prepare bin labels with ranges
    bin_labels_with_ranges = [
        f"{label} ({bin_edges[i]:.2f} - {bin_edges[i+1]:.2f})"
        for i, label in enumerate(pivot_table_percent.columns)
    ]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create stacked bar plot
    bottom = pd.Series([0] * len(pivot_table_percent.index), index=pivot_table_percent.index)
    bar_containers = []
    for category, bin_label in zip(pivot_table_percent.columns, bin_labels_with_ranges):
        bar = ax.bar(pivot_table_percent.index, pivot_table_percent[category], 
                     label=bin_label, bottom=bottom, color=colors[category])
        bar_containers.append(bar)
        
        # Add percentage text to each bar
        for rect, pct in zip(bar, pivot_table_percent[category]):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + height / 2, 
                    f"{pct:.1f}%", ha="center", va="center", fontsize=8)
        
        bottom += pivot_table_percent[category]

    # Customize the plot
    ax.set_xlabel(tag_2, labelpad=10)
    ax.set_ylabel("Percentage")
    ax.set_title(f"'{tag_2}' vs '{tag_1}' (binned) {tags}") #  as Stacked Percentages

    # Adjust layout to increase space for the legend
    ax.legend(title=tag_1 + " (binned)", loc="upper left", bbox_to_anchor=(1, 1)) # + " (Range)"
    plt.subplots_adjust(right=0.75)

    # Show the plot
    plt.show()