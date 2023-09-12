#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>


class dfThermo
{
    // private data members
    const std::string mechanism_file;
    std::string thermo_coeff_file;

    // private member functions
    void readCoeffs(std::ifstream& inputf, int dimension, std::vector<std::vector<double>>& coeffs);
    void initCoeffs(std::ifstream& inputf);

public:
    // public data members
    int num_species;

    // constant value
    double gas_constant;

    // thermo coeffs
    std::vector<std::vector<double>> nasa_coeffs;
    std::vector<std::vector<double>> viscosity_coeffs;
    std::vector<std::vector<double>> thermal_conductivity_coeffs;
    std::vector<std::vector<double>> binary_diffusion_coeffs;

    // TODO: write species
    std::vector<double> mole_fraction;
    std::vector<double> molecular_weights;

    // intermediate variables
    std::vector<double> T_poly;

    // constructor
    dfThermo(std::string mechanism_file);

    // destructor
    ~dfThermo();

    // public member functions
    void calculateTPoly(double T);
    // calculateViscosity must be called earlier than calculateThermoConductivity to calculate T_poly
    double calculateViscosity(double T);
    double calculateThermoConductivity(double T);
    double calculateEnthalpy(double T, double *Y);
    double calculateCp(double T, double *Y);
    double calculateTemperature(double T_init, double target_h, double *Y, double atol = 1e-7, 
            double rtol = 1e-7, int max_iter = 20);

    // getter functions
};