def calculate_power_consumption(gpu, flops):  
    power = gpu["power"]  
    compute_flops_per_second = gpu["compute"] * 1e12
    efficiency_flops_per_joule = compute_flops_per_second / power
    energy_joules = flops / efficiency_flops_per_joule
    energy_kwh = energy_joules / 3600000
    
    return energy_kwh * 1000

def calculate_emissions(gpu, flops, carbon_intensity):
    estimated_power = calculate_power_consumption(gpu, flops)
    emissions = estimated_power * carbon_intensity

    return emissions