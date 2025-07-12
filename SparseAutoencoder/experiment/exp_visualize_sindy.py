import os
import re
import sympy as sp
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from tqdm import tqdm  # <-- NEW

# ------------------ Config ------------------
input_vars = [f'x{i}' for i in range(25)]
x_syms = sp.symbols(input_vars)
output_dir = "save_output"
os.makedirs(output_dir, exist_ok=True)

# ------------------ Parse Equations ------------------
equations = []
with open("sindy_latent_equations.txt", "r") as f:
    lines = f.readlines()

print("📖 Parsing and cleaning symbolic equations...")
for line in tqdm(lines, desc="Cleaning Equations"):
    lhs, rhs = line.strip().split('=')
    z_idx = int(lhs.strip()[2:])  # z_5 → 5

    # Clean RHS
    rhs_cleaned = rhs.strip().replace('[k]', '').replace(' ', '')
    rhs_cleaned = rhs_cleaned.replace('^', '**')
    rhs_cleaned = re.sub(r'(\d)(x)', r'\1*\2', rhs_cleaned)
    rhs_cleaned = re.sub(r'(x\d)(x)', r'\1*\2', rhs_cleaned)
    rhs_cleaned = re.sub(r'(x\d)\(', r'\1*(', rhs_cleaned)
    rhs_cleaned = re.sub(r'([x\d])([+\-])', r'\1 \2 ', rhs_cleaned)

    expr = sp.sympify(rhs_cleaned, locals={v: s for v, s in zip(input_vars, x_syms)})
    equations.append((z_idx, expr))

# ------------------ Generate all possible interaction terms ------------------
interaction_terms = [xi * xj for xi, xj in combinations_with_replacement(x_syms, 2)]
interaction_labels = [f'{str(xi)}*{str(xj)}' for xi, xj in combinations_with_replacement(x_syms, 2)]

# ------------------ Extract Coefficients ------------------
def extract_coeffs(expr):
    def safe_float(val):
        try:
            return float(val.evalf())
        except:
            return 0.0  # fallback to zero

    linear = [safe_float(expr.coeff(xi)) for xi in x_syms]
    inter  = [safe_float(expr.coeff(term)) for term in interaction_terms]
    return linear, inter


# ------------------ Plot and Save ------------------
print("📊 Generating and saving plots...")
for z_idx, expr in tqdm(equations, desc="Plotting & Saving"):
    linear_coeffs, inter_coeffs = extract_coeffs(expr)

    # Linear plot
    plt.figure(figsize=(8, 4))
    plt.bar(range(25), linear_coeffs)
    plt.xticks(range(25), input_vars, rotation=45)
    plt.ylabel("Coefficient")
    plt.title(f"z_{z_idx} — Linear Terms")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"z_{z_idx:04d}_linear.png"))
    plt.close()

    # Interaction plot
    plt.figure(figsize=(16, 5))
    plt.bar(range(len(inter_coeffs)), inter_coeffs)
    plt.xticks(range(len(inter_coeffs)), interaction_labels, rotation=90, fontsize=6)
    plt.ylabel("Coefficient")
    plt.title(f"z_{z_idx} — Interaction Terms")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"z_{z_idx:04d}_interact.png"))
    plt.close()

print(f"\n✅ Saved linear + interaction histograms for {len(equations)} latent units to '{output_dir}/'")
