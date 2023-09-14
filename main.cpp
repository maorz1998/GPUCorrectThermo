#include "dfThermo.h"

int main()
{
    dfThermo thermo("ES80_H2-7-16.yaml");

    std::vector<double> mass_fraction(thermo.num_species, 1/7.0);
    thermo.setMassFraction(mass_fraction);

    // 500K test
    std::cout << "viscosity(500) : " << thermo.calculateViscosity(500) << std::endl;
    std::cout << "thermal_conductivity(500) : " << thermo.calculateThermoConductivity(500) << std::endl;

    // 1500K test
    std::cout << "viscosity(1500) : " << thermo.calculateViscosity(1500) << std::endl;
    std::cout << "thermal_conductivity(1500) : " << thermo.calculateThermoConductivity(1500) << std::endl;
    std::cout << "enthalpy(1500) : " << thermo.calculateEnthalpy(1500) << std::endl;
    std::cout << "Cp(1500) : " << thermo.calculateCp(1500) << std::endl;
    std::cout << "temperature from NewTon's Methods : " << thermo.calculateTemperature(1500, thermo.calculateEnthalpy(1600)) 
        << std::endl;
}