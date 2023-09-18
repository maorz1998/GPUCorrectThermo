#include "dfThermo.h"

int main()
{
    dfThermo thermo("ES80_H2-7-16.yaml");

    std::vector<double> mass_fraction(thermo.num_species, 1/7.0);
    thermo.setMassFraction(mass_fraction);

    // *********single cell test********* //
    // 500K test
    std::cout << "viscosity(500) : " << thermo.calculateViscosity(500) << std::endl;
    std::cout << "thermal_conductivity(500) : " << thermo.calculateThermoConductivity(500) << std::endl;

    // 1500K test
    std::cout << "viscosity(1500) : " << thermo.calculateViscosity(1500) << std::endl;
    std::cout << "thermal_conductivity(1500) : " << thermo.calculateThermoConductivity(1500) << std::endl;
    std::cout << "enthalpy(1500) : " << thermo.calculateEnthalpy(1500) << std::endl;
    std::cout << "Cp(1500) : " << thermo.calculateCp(1500) << std::endl;
    std::cout << "temperature from NewTon's Methods : " << thermo.calculateTemperature(1500, thermo.calculateEnthalpy(1600)) << std::endl;

    // *********field test********* //
    FILE *fp = NULL;
    const char *input_file = "PTHY.txt";
    fp = fopen(input_file, "rb+");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open input file: %s!\n", input_file);
        exit(EXIT_FAILURE);
    }

    // initialize CPU data
    int num_cells, num_species;
    fread(&num_cells, sizeof(int), 1, fp);
    fread(&num_species, sizeof(int), 1, fp);
    // read T & p
    std::vector<double> p(num_cells), T(num_cells), he(num_cells);
    fread(p.data(), sizeof(double), num_cells, fp);
    fread(T.data(), sizeof(double), num_cells, fp);
    fread(he.data(), sizeof(double), num_cells, fp);
    std::vector<std::vector<double>> Y(num_cells);
    for (int i = 0; i < num_cells; i++) {
        Y[i].resize(num_species);
        fread(Y[i].data(), sizeof(double), num_species, fp);
    }
    fclose(fp);

    // initialize GPU data
    double *d_p, *d_T, *d_he;
    
    

    // CPU: correct Thermo loop
    double *T_new = new double[num_cells];
    double *psi = new double[num_cells];
    double *rho = new double[num_cells];
    double *mu = new double[num_cells];
    double *alpha = new double[num_cells];
    for (int i = 0; i < num_cells; i++) {
        thermo.setMassFraction(Y[i]);
        T_new[i] = thermo.calculateTemperature(T[i], he[i]);
        psi[i] = thermo.calculatePsi(T_new[i]);
        rho[i] = thermo.calculateRho(p[i], psi[i]);
        mu[i] = thermo.calculateViscosity(T_new[i]);
        alpha[i] = thermo.calculateThermoConductivity(T_new[i]);
        std::cout << "alpha[" << i << "] : " << alpha[i] << std::endl;
    }
}