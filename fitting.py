import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

import errors as err

def read_data(file_paths, selected_country, start_year, end_year):
    dataframes_list = []

    for path in file_paths:
        file_name = path.split('/')[-1].split('.')[0]
        df = pd.read_csv(path, skiprows=4)
        df = df.rename(columns={'Country Name': 'Country'})
        df = df.set_index('Country')
        df_selected_country = df.loc[selected_country, str(start_year):str(end_year)].transpose().reset_index()
        df_selected_country = df_selected_country.rename(columns={'index': 'Year', selected_country: file_name})
        dataframes_list.append(df_selected_country)

    # Concatenate all DataFrames based on the 'Year' column
    result_df = pd.concat(dataframes_list, axis=1)

    # Replace null values with the mean of each column
    result_df = result_df.apply(lambda col: col.fillna(col.mean()))

    return result_df

def fit_and_plot_growth_model(data_df, label, output_filename):
    def exp_growth(t, scale, growth):
        """Computes exponential function with scale and growth as free parameters"""
        return scale * np.exp(growth * t)

    data_df["Year"] = pd.to_numeric(data_df["Year"], errors='coerce')
    data_df[label] = pd.to_numeric(data_df[label], errors='coerce')

    initial_guess = [1.0, 0.02]
    popt, pcovar = opt.curve_fit(exp_growth, data_df["Year"], data_df[label], p0=initial_guess, maxfev=10000)
    print("Fit parameters:", popt)

    # Create a new column with the fitted values
    data_df["pop_exp"] = exp_growth(data_df["Year"], *popt)

    # Plot
    plt.figure()
    plt.plot(data_df["Year"], data_df[label], label="data")
    plt.plot(data_df["Year"], data_df["pop_exp"], label="fit")
    plt.legend()
    plt.title("Data Fit attempt")
    plt.show()

    # Call function to calculate upper and lower limits with extrapolation
    # Create extended year range
    years = np.linspace(data_df["Year"].min(), data_df["Year"].max() + 10)
    pop_exp_growth = exp_growth(years, *popt)
    sigma = err.error_prop(years, exp_growth, popt, pcovar)
    low = pop_exp_growth - sigma
    up = pop_exp_growth + sigma

    plt.figure()
    plt.title(f"{label} of Brazil in 2030")
    plt.plot(data_df["Year"], data_df[label], label="data")
    plt.plot(years, pop_exp_growth, label="fit")
    # Plot error ranges with transparency
    plt.fill_between(years, low, up, alpha=0.3, color="y", label="95% Confidence Interval")
    plt.legend(loc="upper left")
    plt.xlabel("Year")
    plt.ylabel(label)
    # Set the dpi parameter to 300 when saving the plot
    plt.savefig(f"{output_filename}.png", dpi=300)
    plt.show()

    # Predict future values
    pop_2030 = exp_growth(np.array([2030]), *popt)
    # Assuming you want predictions for the next 10 years
    sigma_2030 = err.error_prop(np.array([2030]), exp_growth, popt, pcovar)
    print(f"{label} in")
    print("2030:", exp_growth(2030, *popt) / 1.0e6, "Mill.")

    # For next 10 years
    print(f"{label} in")
    for year in range(2024, 2034):
        print(f"{label} in", year)
        print("2030:", exp_growth(year, *popt) / 1.0e6, "Mill.")

# Example usage:

# Cereal yield
selected_country = "Brazil"
start_year = 1960
end_year = 2021
file_paths = ['Cereal yield.csv']
Cereal_yield = read_data(file_paths, selected_country, start_year, end_year)
fit_and_plot_growth_model(Cereal_yield, "Cereal yield", "Cereal_yield")

# Forest area
selected_country = "Brazil"
start_year = 1990
end_year = 2021
file_paths = ['Forest area.csv']
Forest_area = read_data(file_paths, selected_country, start_year, end_year)
fit_and_plot_growth_model(Forest_area, "Forest area", "Forest_area")

# Agricultural land
selected_country = "Brazil"
start_year = 1990
end_year = 2021
file_paths = ['Agricultural land.csv']
Crop_production_df = read_data(file_paths, selected_country, start_year, end_year)
fit_and_plot_growth_model(Crop_production_df, "Agricultural land", "Agricultural_land")
