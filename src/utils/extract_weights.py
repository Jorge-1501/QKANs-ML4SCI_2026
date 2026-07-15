# extract_weights.py
import numpy as np
from numpy.polynomial.chebyshev import chebfit, cheb2poly
import os
import torch
from kan import KAN
import src.utils.workspace as workspace

def evaluate_isolated_edges(model, layer_index, input_index, output_index, x_vals):
    """
    Evaluate the output of a specific layer and output node while varying only one input variable at a time. This allows us 
    to isolate the effect of that input variable on the output, which is crucial for accurately fitting Chebyshev polynomials 
    to the model's behavior.
    """
    layer_width = model.width[layer_index]
    in_dim = layer_width[0] if isinstance(layer_width, list) else layer_width
    n = len(x_vals)

    # Put almost every variable into 0 to extract coefficients
    x_zero = torch.zeros((n,in_dim), dtype=torch.float32)
    
    x_var = torch.zeros((n,in_dim), dtype=torch.float32)
    x_var[:, input_index] = torch.tensor(x_vals, dtype=torch.float32)

    def layer_forward(x_in):
        # Symbolic part
        try:
            symbolic = model.symbolic_fun[layer_index](x_in)
            x_out = symbolic[0] if isinstance(symbolic, tuple) else symbolic
        except Exception:
            x_out = torch.zeros((n, model.width[layer_index + 1]), dtype=torch.float32)

        # Bias nad scale 
        # hasattr asks if the model has the attribute of the second argument
        if hasattr(model, "node_bias") and model.node_bias is not None and len(model.node_bias) > layer_index:
            x_out += model.node_bias[layer_index]
        if hasattr(model, "edge_scale") and model.edge_scale is not None and len(model.edge_scale) > layer_index:
            x_out *= model.edge_scale[layer_index]
        return x_out
    
    with torch.no_grad():
        y_var = layer_forward(x_var)[:, output_index].numpy()
        y_zero = layer_forward(x_zero)[:, output_index].numpy()

    y_var = np.nan_to_num(y_var, nan=0.0, posinf=0.0, neginf=0.0)
    y_zero = np.nan_to_num(y_zero, nan=0.0, posinf=0.0, neginf=0.0)

    # Global correction to distribute the bias and scale effects
    f_x = y_var - y_zero + (y_zero / in_dim)
    return f_x

def save_symbolic_report(save_path, coefs_dict, degrees_dict):
    """
    Generate a text file with the mathematical formulas of the 
    polynomials fitted for each variable.
    """
    with open(save_path, "w") as f:
        f.write("==================================================\n")
        f.write("SYMBOLIC APPROXIMATION REPORT (CHEBYSHEV)\n")
        f.write("==================================================\n\n")
        
        for name, coefs in coefs_dict.items():
            deg = degrees_dict[name]
            # Convert from Chebyshev basis to standard polynomial basis (1, x, x^2...)
            poly_coeffs = cheb2poly(coefs)
            
            f.write(f"VARIABLE: {name}\n")
            f.write(f"Polynomial degree (d): {deg}\n")
            f.write(f"Chebyshev coefficients (c_i): {coefs.tolist()}\n")
            
            # Construct a readable formula: f(x) = a + bx + cx^2...
            formula = "f(x) = "
            terms = []
            for i, c in enumerate(poly_coeffs):
                if abs(c) < 1e-5: continue # Omit insignificant terms
                if i == 0: terms.append(f"{c:.6f}")
                elif i == 1: terms.append(f"({c:.6f} * x)")
                else: terms.append(f"({c:.6f} * x^{i})")
            
            f.write(" + ".join(terms).replace("+ -", "- ") + "\n")
            f.write("-" * 50 + "\n\n")

def extract_chebyshev_weights(CONFIG, force=False):

    print("\n" + "="*40)
    print("Automated Chebyshev extraction")
    print("="*40)

    # load the model to extract the functions
    final_model_prefix = os.path.join(CONFIG['final_model_path'], "05_final")
    model_path = final_model_prefix
    model = KAN.loadckpt(model_path)
    model.eval()

    # Quantum domain: x values in the range [-1, 1]
    x_vals = np.linspace(-1, 1, 1000)

    # Extracting the functions for each input variable by isolating them
    y_n = evaluate_isolated_edges(model, layer_index=0, input_index=0, output_index=0, x_vals=x_vals)
    y_q = evaluate_isolated_edges(model, layer_index=0, input_index=1, output_index=0, x_vals=x_vals)
    y_z = evaluate_isolated_edges(model, layer_index=0, input_index=2, output_index=0, x_vals=x_vals)
    y_dr = evaluate_isolated_edges(model, layer_index=0, input_index=3, output_index=0, x_vals=x_vals)

    y_out = evaluate_isolated_edges(model, layer_index=1, input_index=0, output_index=0, x_vals=x_vals)

    # degree 4 polynomials to fit the functions
    deg_hidden = 1
    deg_output = 1

    # Calculate Chebyshev coefficients (Degree d=4)
    w_n = chebfit(x_vals, y_n, deg=deg_hidden)
    w_q = chebfit(x_vals, y_q, deg=deg_hidden)
    w_z = chebfit(x_vals, y_z, deg=deg_hidden)
    w_dr = chebfit(x_vals, y_dr, deg=deg_hidden)
    w_out = chebfit(x_vals, y_out, deg=deg_output)

    print(f"Initial weights for n: {w_n}")
    print(f"Initial weights for q: {w_q}")
    print(f"Initial weights for z: {w_z}")
    print(f"Initial weights for dr: {w_dr}")
    print(f"Initial weights for output: {w_out}")

    # Saving coefficients to a .npy file
    np.save(CONFIG["coef_n_path"], w_n)
    np.save(CONFIG["coef_q_path"], w_q)
    np.save(CONFIG["coef_z_path"], w_z)
    np.save(CONFIG["coef_dr_path"], w_dr)
    np.save(CONFIG["coef_out_path"], w_out)

    print(f"Chebyshev coefficients saved to '{CONFIG['polynomial_weights_dir']}' directory.")

    coefs_dict = {
        "n": w_n,
        "q": w_q,
        "z": w_z,
        "dr": w_dr,
        "Output_Layer": w_out
    }

    degrees_dict = {
        "n": deg_hidden,
        "q": deg_hidden,
        "z": deg_hidden,
        "dr": deg_hidden,
        "Output_Layer": deg_output
    }

    report_path = CONFIG["Chebyshev_coefficients_path"]
    save_symbolic_report(report_path, coefs_dict, degrees_dict)
    print(f"Symbolic report generated at: {report_path}")