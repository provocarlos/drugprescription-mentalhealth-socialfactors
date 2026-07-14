import plotly.express as px
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import plotly.graph_objects as go

VAR_NAMES_MAPPING = {
        "items_per_1000": "Items per 1,000 Population",
        "quantity_per_1000": "Quantity per 1,000 Population",
        "imd_quintile": "IMD Quintile",
        "imd_score": "IMD Score",
        "spend_per_1000": "Spending (£) per 1,000 Population",
        "cost_per_item": "Cost (£) per Item",
        'Antidepressants': 'Antidepressant Drugs',
        'ADHD': 'CNS Stimulants and Drugs for ADHD',
        'Anxiolytics': 'Anxiolytic Drugs',
        'cluster': 'K-means Cluster'
    }

def plot_time_series_longi_trends_for_drug(dataframe, y="items_per_1000",):
    fig_trend = px.line(dataframe, x='date', y=y, color='drug_category',
                        title='<b>Longitudinal Prescription Trends (2010-2024)</b><br><i>Items per 1,000 Population (Normalised)</i>',
                        labels={y: VAR_NAMES_MAPPING[y], 'date': 'Year',
                                'drug_category': 'Condition Proxy'},
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        template='plotly_white')

    # Add "Event Lines" to visualy correlate with diagnostic changes
    events = [
        # ('2013-05-18', 'DSM-5 Released', 'Dash'),
        # ('2018-03-14', 'NICE ADHD Guidelines Update', 'Dot'),
        # ('2020-03-23', 'COVID-19 Lockdown', 'Solid'),
        ('2022-03-01', 'DSM-5-TR Released', 'Dash')
    ]

    for date, label, style in events:
        fig_trend.add_vline(x=date, line_width=1, line_dash="dash", line_color="grey")
        fig_trend.add_annotation(x=date, y=dataframe[y].max(),
                                 text=label, showarrow=False, yshift=10, textangle=-90)

    fig_trend.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    fig_trend.show()


def plot_line_socioeconomic_gradients(data, title):
    fig_social_adhd = px.line(
        data,
        x='date',
        y='items_per_1000',
        color='imd_quintile',
        title=title,
        labels={'items_per_1000': 'Items per 1,000 People', 'imd_quintile': 'Deprivation Quintile'},
        color_discrete_sequence=px.colors.sequential.Viridis_r,
        template='plotly_white'
    )
    fig_social_adhd.update_layout(legend=dict(orientation="h", y=-0.2))
    fig_social_adhd.show()

def plot_regional_variations_box_and_scatter(data, title, colour = '#1f77b4'):
    fig_region_adhd = px.box(
        data,
        y='items_per_1000',
        points="all",  # Show individual CCG dots
        color_discrete_sequence=[colour],  # Single color for ADHD
        title=title,
        labels={'items_per_1000': 'Items / 1,000 People (Annual Avg)'},
        template='plotly_white'
    )
    fig_region_adhd.show()

def plot_scatter_with_lobf(data, title,x,y, colour = '#1f77b4'):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=data,
        x="imd_score",
        y="items_per_1000",
        alpha=0.3,
        color=colour
    )
    sns.regplot(
        data=data,
        x="imd_score",
        y="items_per_1000",
        scatter=False,
        line_kws={'linewidth': 3, 'color': colour}
    )
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.show()


def plot_ridge_distribution_by_imd(df, drug, plot_var='items_per_1000', output_dir="util/vis/ridgeplots"):
    """
    Creates a ridge plot showing the distribution of a specified variable
    (e.g., items_per_1000) for a given drug category across different IMD quintiles.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    imd_order = ['Most Deprived (1)', '2', '3', '4', 'Least Deprived (5)']

    # Filter data for current substance
    substance_data = df[df['drug_category'] == drug].copy()

    if substance_data.empty:
        print(f"No data available for drug category: {drug}. Skipping plot.")
        return

    # Get unique quintiles
    quintiles = [q for q in imd_order if q in substance_data['imd_quintile'].values]

    # Create a beautiful color gradient (from light to dark)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(quintiles)))

    # Create figure
    fig, axes = plt.subplots(len(quintiles), 1, figsize=(14, len(quintiles) * 1.8),
                             sharex=True)

    # Find global min/max for consistent x-axis
    all_data = substance_data[plot_var]
    global_min = all_data.min()
    global_max = all_data.max()
    x_range = np.linspace(global_min - (global_max - global_min) * 0.05,
                          global_max + (global_max - global_min) * 0.05, 300)

    # Create ridge plot
    for idx, quintile in enumerate(quintiles):
        ax = axes[idx]
        ax.set_facecolor('white')

        # Get data for this quintile
        quintile_data = substance_data[substance_data['imd_quintile'] == quintile][plot_var]

        # Calculate median
        median_value = quintile_data.median()

        # Calculate KDE with bandwidth adjustment
        kde = stats.gaussian_kde(quintile_data, bw_method=0.3)
        density = kde(x_range)

        # Normalize density for better visual appeal
        density_normalized = density / density.max() * 0.8

        # Find the y-value at the median on the density curve
        median_idx = np.argmin(np.abs(x_range - median_value))
        median_y = density_normalized[median_idx]

        # Create gradient fill
        ax.fill_between(x_range, density_normalized, alpha=0.6,
                        color=colors[idx], linewidth=0)

        # Add outline
        ax.plot(x_range, density_normalized, linewidth=1.5,
                color=colors[idx], alpha=1)

        # Add subtle grid on the ridge
        ax.plot(x_range, density_normalized, linewidth=0.2,
                alpha=0.4, linestyle='-')

        # Customize individual subplot
        ax.text(0.02, 0.85, quintile, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='center',
                color=colors[idx])

        # Add median line
        ax.plot([median_value, median_value], [0, median_y],
                color='#e74c3c', linewidth=2.5, linestyle='--',
                alpha=0.9, zorder=10)

        # Add median label
        ax.text(median_value, median_y - 0.5, f'{median_value:.1f}',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                color='#e74c3c',
                zorder=11,
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='white',
                          edgecolor='#e74c3c',
                          linewidth=2,
                          alpha=0.95))

        # Aesthetics
        ax.set_yticks([])
        ax.set_xlim()
        ax.tick_params(axis='x', colors='#333333', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('#666666')

    # Overall title and labels
    fig.suptitle(f'Distribution of {VAR_NAMES_MAPPING[drug]} Prescriptions by Socioeconomic Status',
                 fontsize=16, fontweight='bold', y=0.99, color='#2c3e50')

    axes[-1].set_xlabel(VAR_NAMES_MAPPING[plot_var], fontsize=12,
                        color='#2c3e50', fontweight='600')

    # Add subtitle
    fig.text(0.5, 0.94, 'IMD Quintile (Most to Least Deprived). Median indicated by dashed line.',
             ha='center', fontsize=14, color='#7f8c8d', style='italic')

    plt.subplots_adjust(hspace=-0.1)  # Overlap the plots slightly
    # Save the plot
    safe_filename = drug.replace('/', '_').replace('\\', '_')
    output_path = f'{output_dir}/ridgeplot_{safe_filename}.png'
    #plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()


def plot_interactive_map(geojson, df, years, drug_categories):
    # Create figure
    fig = go.Figure()

    # Store all year-category combinations as traces for reactive dropdowns
    trace_map = {}
    trace_idx = 0

    for year in years:
        for category in drug_categories:
            # Filter data for the specific year and category
            filtered_data = df[(df['year'] == year) &
                               (df['drug_category'] == category)]

            # Only add trace if there's data
            if len(filtered_data) > 0:
                color_min = filtered_data['items_per_1000'].min()
                color_max = filtered_data['items_per_1000'].max()

                # Calculate range for colors
                color_range = color_max - color_min
                if color_range == 0:
                    color_range = 1
                color_min = max(0, color_min - 0.05 * color_range)
                color_max = color_max + 0.05 * color_range

                # Add Choropleth trace
                fig.add_trace(go.Choropleth(
                    geojson=geojson,
                    locations=filtered_data['row_id'],
                    z=filtered_data['items_per_1000'],
                    featureidkey="properties.code",
                    colorscale="thermal",
                    zmin=color_min,
                    zmax=color_max,
                    marker_line_width=0.5,
                    marker_line_color='white',
                    colorbar=dict(title="items per<br>1,000 pop"),
                    hovertemplate='%{customdata[0]}<br>' +
                                'Items per 1,000: %{z:.2f}<br>' +
                                'Total List Size: %{customdata[1]:,d}<br>' +
                                'IMD Rank: %{customdata[2]}/106<br>' +
                                '<extra></extra>',
                    customdata=filtered_data[['row_name', 'total_list_size', 'imd_rank']],
                    visible=(year == years[0] and category == drug_categories[0]),
                    name=f"{year} - {category}",
                ))

                trace_map[(year, category)] = trace_idx
                trace_idx += 1

    # Independent dropdowns for Year and Drug Category

    year_buttons = []
    for year in years:
        visibility = []
        for y in years:
            for c in drug_categories:
                if (y, c) in trace_map:
                    visibility.append(y == year and c == drug_categories[0])

        year_buttons.append(
            dict(
                label=str(year),
                method='update',
                args=[{'visible': visibility},
                      {'title': f"Year: {year}, Category: {drug_categories[0]}"}]
            )
        )

    category_buttons = []
    for category in drug_categories:
        visibility = []
        for y in years:
            for c in drug_categories:
                if (y, c) in trace_map:
                    visibility.append(y == years[0] and c == category)

        category_buttons.append(
            dict(
                label=category,
                method='update',
                args=[{'visible': visibility},
                      {'title': f"Year: {years[0]}, Category: {category}"}]
            )
        )

    # Update layout with separate dropdown menus
    fig.update_layout(
        title=f"Year: {years[0]}, Category: {drug_categories[0]}",
        template="plotly_white",
        height=450,
        updatemenus=[
            dict(
                buttons=year_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
                active=0
            ),
            dict(
                buttons=category_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.35,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
                active=0
            ),
        ],
        annotations=[
            dict(text="Year:", showarrow=False, x=0.05, y=1.18,
                 xref="paper", yref="paper", align="left"),
            dict(text="Drug Category:", showarrow=False, x=0.28, y=1.18,
                 xref="paper", yref="paper", align="left")
        ],
        geo=dict(
            fitbounds="locations",
            visible=True
        )
    )
    fig.update_geos(
        resolution=50,
        showcoastlines=True, coastlinecolor="RebeccaPurple",
        showocean=True, oceancolor="#ADD8E6",
        showlakes=True, lakecolor="#2EBEFC",
        showrivers=True, rivercolor="#1FDBD8",

    )

    fig.show()


def plot_categorical_choropleth(data, geo_df, color_column, title=None):
    """
    Create a categorical choropleth map using Plotly.
    
    Parameters:
    -----------
    data : GeoDataFrame
        Geographic data with geometry column
    color_column : str
        Column name containing categorical values
    title : str, optional
        Map title
    """
    # Convert cluster column to string for categorical mapping
    data = data.copy()
    data[color_column] = data[color_column].astype(str)
    
    fig = px.choropleth(
        data,
        geojson=geo_df,
        locations=data['row_id'],
        featureidkey="properties.code",
        color=color_column, 
        title=title,
        hover_name="row_name",
        hover_data={
            'cluster': True,
        },
        template="plotly_white"
    )
    
    fig.update_geos(
        fitbounds="locations", 
        visible=False,
        resolution=50,
        showcoastlines=True, 
        coastlinecolor="RebeccaPurple",
        showocean=True, 
        oceancolor="#ADD8E6",
        showlakes=True, 
        lakecolor="#2EBEFC",
        showrivers=True, 
        rivercolor="#1FDBD8"
    )
    
    fig.update_layout(
        template="plotly_white",
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        height=450
    )
    
    return fig.show()

def plot_cluster_scater(df):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
    data=df,
    x='imd_score',
    y='items_per_1000',
    hue='cluster',
    palette='viridis',
    s=100,
    style='cluster')

    plt.title('K-Means Clustering of NHS Regions (ADHD Prescribing)')
    plt.xlabel('Deprivation Score')
    plt.ylabel('Items per 1,000')
    plt.legend(title='Cluster Group')
    plt.show()

