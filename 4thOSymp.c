#include <cmath>
#include <iostream>

using namespace std;

const double total_time = 5e3;

// Parameters for the potential.
const double sigma = 1.0;
const double sigma6 = pow(sigma, 6.0);
const double epsilon = 1.0;
const double four_epsilon = 4.0 * epsilon;

// Constants used in the composition method.
const double alpha = 1.0 / (2.0 - cbrt(2.0));
const double beta = 1.0 - 2.0 * alpha;


static double force(double q, double& potential);

static void verlet(double dt,
 double& q, double& p,
 double& force, double& potential);

static void composition_method(double dt,
 double& q, double& p,
 double& f, double& potential);


int main() {
  const double q0 = 1.5, p0 = 0.1;
  double potential;
  const double f0 = force(q0, potential);
  const double total_energy_exact = p0 * p0 / 2.0 + potential;

  for (double dt = 1e-2; dt <= 5e-2; dt *= 1.125) {
    const long steps = long(total_time / dt);

    double q = q0, p = p0, f = f0;
    double total_energy_average = total_energy_exact;

    for (long step = 1; step <= steps; ++step) {
      composition_method(dt, q, p, f, potential);
      const double total_energy = p * p / 2.0 + potential;
      total_energy_average += total_energy;
    }

    total_energy_average /= double(steps);

    const double err = fabs(total_energy_exact - total_energy_average);
    cout << log10(dt) << "\t"
    << log10(err) << endl;
  }

  return 0;
}

double force(double q, double& potential) {
  const double r2 = q * q;
  const double r6 = r2 * r2 * r2;
  const double factor6  = sigma6 / r6;
  const double factor12 = factor6 * factor6;

  potential = four_epsilon * (factor12 - factor6);
  return -four_epsilon * (6.0 * factor6 - 12.0 * factor12) / r2 * q;
}

void verlet(double dt,
  double& q, double& p,
  double& f, double& potential) {
  p += dt / 2.0 * f;
  q += dt * p;
  f = force(q, potential);
  p += dt / 2.0 * f;
}

void composition_method(double dt,
  double& q, double& p,
  double& f, double& potential) {
  verlet(alpha * dt, q, p, f, potential);
  verlet(beta * dt, q, p, f, potential);
  verlet(alpha * dt, q, p, f, potential);
}