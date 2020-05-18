from enum import Enum


class FundamentalNat(Enum):
    hbar = 1

class FundamentalSI(Enum):
    hbar = 1.0545718E-25    # µm^2 kg / ms

class MassNat(Enum):
    neutrino =                2      # eV / c^2
    muonNeutrino =      170_000      # eV / c^2
    electron =          510_998.950  # eV / c^2
    muon =          105_700_000      # eV / c^2
    tau =         1_776_860_000      # eV / c^2
    rubidium87 = 80_955_389_107.0556 # eV / c^2

class MassSI(Enum):
    neutrino =     3.56532384326E-36    # kg
    muonNeutrino = 3.03052526677E-31    # kg
    electron =     9.10938370157E-30    # kg
    muon =         1.88427365116E-28    # kg
    tau =          3.16754066206E-27    # kg
    rubidium87 =   1.44316089511E-25    # kg
