def calculate_power_consumption(gpu, flops):  
    if not gpu or gpu.get("power") is None or gpu.get("compute") is None or gpu["power"] == 0:
        return 0.0  # or 0.0
    power = gpu["power"]  
    compute_flops_per_second = gpu["compute"] * 1e12
    efficiency_flops_per_joule = compute_flops_per_second / power
    energy_joules = flops / efficiency_flops_per_joule
    energy_kwh = energy_joules / 3600000
    
    return energy_kwh * 1000

def calculate_emissions(gpu, flops, carbon_intensity):
    if gpu is None or carbon_intensity is None:
        return 0.0  # or 0.0
    estimated_power = calculate_power_consumption(gpu, flops)
    emissions = estimated_power * carbon_intensity

    return emissions