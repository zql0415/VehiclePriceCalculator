import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

def run_cohort_value_retention_analysis():
    print("\n=== Cohort-Based Value Retention Analysis ===")

    df = pd.read_csv("data/car_prices.csv")

    df = df.rename(columns={'make': 'brand', 'odometer': 'mileage', 'body': 'body_type'})
    df['sale_year'] = df['saledate'].astype(str).str[11:15]
    df['sale_year'] = pd.to_numeric(df['sale_year'], errors='coerce').astype('Int64')
    df['car_age'] = df['sale_year'] - df['year']
    df = df.dropna(subset=['mmr', 'sellingprice', 'car_age', 'year'])

    df['value_retention_ratio'] = (df['sellingprice'] / df['mmr']).round(4)

    df['cohort'] = df['year']
    cohort_data = df.groupby(['cohort', 'car_age'])['value_retention_ratio'].agg(['mean', 'count']).reset_index()
    cohort_pivot = cohort_data.pivot(index='cohort', columns='car_age', values='mean')
    count_pivot = cohort_data.pivot(index='cohort', columns='car_age', values='count')

    valid_cohorts = count_pivot.index[count_pivot.sum(axis=1) >= 100]
    valid_ages = count_pivot.columns[count_pivot.sum() >= 100]
    filtered_pivot = cohort_pivot.loc[valid_cohorts, valid_ages]

    plt.figure(figsize=(12, 8))
    sns.heatmap(filtered_pivot, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Vehicle Value Retention by Manufacturing Year Cohort')
    plt.xlabel('Vehicle Age (years)')
    plt.ylabel('Manufacturing Year Cohort')
    plt.tight_layout()
    plt.savefig('dataviz/cohort_retention_heatmap.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    for year in sorted(filtered_pivot.index)[-5:]:
        if year in filtered_pivot.index:
            plt.plot(filtered_pivot.columns, filtered_pivot.loc[year], marker='o', label=f'Year {year}')
    plt.title('Value Retention Over Time by Manufacturing Year')
    plt.xlabel('Vehicle Age (years)')
    plt.ylabel('Value Retention Ratio')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('dataviz/cohort_retention_lines.png')
    plt.close()

    avg_retention = filtered_pivot.mean(axis=0)
    print("\nAverage Value Retention Ratio by Vehicle Age:")
    for age, retention in avg_retention.items():
        print(f"Age {age} years: {retention:.2f}")

    best_years = filtered_pivot.mean(axis=1).sort_values(ascending=False)
    print("\nTop Cohort Years by Retention:")
    for year, retention in best_years.head(5).items():
        print(f"Year {year}: {retention:.2f}")

    df['meter_value_ratio'] = df['mileage'] / df['sellingprice']
    model_summary = df.groupby(['brand', 'model']).agg(
        avg_retention=('value_retention_ratio', 'mean'),
        avg_meter_value=('meter_value_ratio', 'mean'),
        count=('value_retention_ratio', 'count')
    ).reset_index()
    filtered_models = model_summary[model_summary['count'] >= 30]
    top_models = filtered_models.sort_values(by='avg_retention', ascending=False)
    print("\nTop 10 Value-Retaining Models:")
    print(top_models.head(10))

    fixed_ages = [3, 4, 5, 6, 7, 8]
    plt.figure(figsize=(14, 6))
    for age in fixed_ages:
        age_df = df[df['car_age'] == age].groupby('cohort')['value_retention_ratio'].mean().reset_index()
        plt.plot(age_df['cohort'], age_df['value_retention_ratio'], marker='o', label=f'Age {age}')
    plt.title('Value Retention at Fixed Ages by Cohort')
    plt.xlabel('Cohort (Production Year)')
    plt.ylabel('Value Retention Ratio')
    plt.axhline(y=1.0, color='red', linestyle='--', label='MMR Baseline')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("dataviz/cohort_fixed_age_lines.png")
    plt.close()
    
    model_df = df[df['car_age'] <= 10].copy()
    model_df['car_age'] = model_df['car_age'].astype(int)
    model_df['cohort'] = model_df['cohort'].astype(int)
    regression = smf.ols('value_retention_ratio ~ car_age + cohort', data=model_df).fit()

    print("\nRegression Summary: Value Retention ~ Car Age + Cohort")
    print(regression.summary())

    if 'body_type' in df.columns:
        body_retention = df.groupby('body_type')['value_retention_ratio'].agg(['mean', 'count']).reset_index()
        body_retention = body_retention[body_retention['count'] >= 30].sort_values('mean', ascending=False)
        print("\nAverage Value Retention by Vehicle Type:")
        print(body_retention)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=body_retention, x='body_type', y='mean')
        plt.axhline(y=1.0, color='red', linestyle='--')
        plt.title('Average Value Retention by Vehicle Type')
        plt.xlabel('Vehicle Type')
        plt.ylabel('Retention Ratio')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("dataviz/cohort_retention_by_type.png")
        plt.close()

if __name__ == "__main__":
    run_cohort_value_retention_analysis()
