# src/architectures/hep_kan.py
import torch
import copy
import inspect
import matplotlib
import matplotlib.pyplot as plt
import sympy
from kan import KAN
import os
import yaml
import numpy as np
import gc
import io

class HEPKAN(KAN):
    def __init__(self, *args, **kwargs):
        """
        Initializes the official KAN model from the library.
        Maintains all original functionality but allows method overrides.
        """
        super().__init__(*args, **kwargs)

    def prune_input(self, threshold=1e-2, active_inputs=None, log_history=True):
        """
        Override de MultKAN.prune_input.

        Method to prune the input layer of the model based on a threshold. 
        If active_inputs is provided, it will use those inputs instead of 
        calculating them based on the threshold.

        Note: This method is a workaround for a bug in the original pykan library.
        Bug: The bug is that when reconstructing the sub-model, MultKAN.prune_input 
        passes `base_fun=self.base_fun` (the already instantiated module, e.g., nn.SiLU())
        instead of `base_fun=self.base_fun_name` (the string 'silu'). Since 
        `nn.SiLU() == 'silu'` is False, the new model ends up with 
        `base_fun_name` pointing to a nn.Module object instead of a 
        string, which breaks YAML serialization in saveckpt (errors 
        related to "handling silu") and causes input pruning not to 
        survive saving/loading.

        This version is identical to the original except for using
        `self.base_fun_name` when reconstructing the model, just like 
        `prune()`/`prune_node()` already do correctly.

        args:
            - threshold: float
                The threshold for pruning. Inputs with scores below 
                this value will be pruned.
            - active_inputs: list or None
                If provided, this list of input indices will be used 
                instead of calculating them based on the threshold.
            - log_history: bool
                If True, logs the pruning operation in the model's history.
        returns:
            - model: HEPKAN
                A new HEPKAN model with the pruned input layer.
        """
        if active_inputs is None:
            self.attribute()
            input_score = self.node_scores[0]
            input_mask = input_score > threshold
            print('keep:', input_mask.tolist())
            input_id = torch.where(input_mask == True)[0]
        else:
            input_id = torch.tensor(active_inputs, dtype=torch.long).to(self.device)

        model2 = HEPKAN(
            copy.deepcopy(self.width), grid=self.grid, k=self.k,
            base_fun=self.base_fun_name,  # <-- FIX: string, no el módulo
            mult_arity=self.mult_arity, ckpt_path=self.ckpt_path,
            auto_save=True, first_init=False, state_id=self.state_id,
            round=self.round
        ).to(self.device)
        model2.load_state_dict(self.state_dict())

        model2.act_fun[0] = model2.act_fun[0].get_subset(input_id, torch.arange(self.width_out[1]))
        model2.symbolic_fun[0] = self.symbolic_fun[0].get_subset(input_id, torch.arange(self.width_out[1]))

        model2.cache_data = self.cache_data
        model2.acts = None

        model2.width[0] = [len(input_id), 0]
        model2.input_id = input_id

        if log_history:
            self.log_history('prune_input')
            model2.state_id += 1

        return model2

    def log_history(self, method_name):
        """
        Bypasses redundant IO of pykan.
        Prevents the library from attempting to write logs and automatic checkpoints
        in every operation, optimizing RAM and time in the classic KAN.
        """
        return 

    def plot(self, folder="./figures", save_path=None, beta=3, metric='backward', 
             scale=0.5, tick=False, sample=False, in_vars=None, out_vars=None, 
             title=None, varscale=1.0, edge_dpi=150, thumb_dpi=50):
        """
        plot KAN - visualizes the network architecture and activations. Optmizations
        based on the original pykan.plot() to avoid redundant IO and improve performance.
        
        Changes compared to the original pykan version:
          1. Filter pruned edges: completely skips rendering any edge with symbolic_mask == 0
                and numeric_mask == 0 (O(E_active) instead of O(E_total)).
          2. Force Backend headless (Agg) and a SINGLE pair (fig, ax) recycled for
             all edge subplots: cleared with ax.clear() instead of creating/destroying a 
             Figure per edge.
          3. A single render per edge: it is drawn once at high resolution,
             saved to disk, and the thumbnail for the global graph is derived from the same
             already rasterized buffer (RGBA in-memory) instead of triggering a second
             full savefig() pass.
          4. Thumbnails stored as uint8 (not float64) to minimize RAM.
          5. Dynamic axes/ranges calculated by the B-spline, with no real extra cost.

        Args:
        -----
            folder : str
                the folder to store pngs
            beta : float
                positive number. control the transparency of each activation. transparency = tanh(beta*l1).
            mask : bool
                If True, plot with mask (need to run prune() first to obtain mask). If False (by default), plot all activation functions.
            mode : bool
                "supervised" or "unsupervised". If "supervised", l1 is measured by absolution value (not subtracting mean); if "unsupervised", l1 is measured by standard deviation (subtracting mean).
            scale : float
                control the size of the diagram
            in_vars: None or list of str
                the name(s) of input variables
            out_vars: None or list of str
                the name(s) of output variables
            title: None or str
                title
            varscale : float
                the size of input variables
            edge_dpi : int
                the dpi of edge figures
            thumb_dpi : int
                the dpi of thumbnail figures
            
        Returns:
        --------
            Figure
            
        Example
        -------
        >>> # see more interactive examples in demos
        >>> model = KAN(width=[2,3,1], grid=3, k=3, noise_scale=1.0)
        >>> x = torch.normal(0,1,size=(100,2))
        >>> model(x) # do a forward pass to obtain model.acts
        >>> model.plot()
        """
        matplotlib.use('Agg', force=True)  # headless: never ever tries to talk to an X11 display via SSH
        global Symbol
        kan_file_path = inspect.getfile(KAN)
        kan_dir = os.path.dirname(os.path.abspath(kan_file_path))
 
        if not self.save_act:
            print('cannot plot since data are not saved. Set save_act=True first.')
 
        if self.acts is None:
            if self.cache_data is None:
                raise Exception('model hasn\'t seen any data yet.')
            self.forward(self.cache_data)
 
        if metric == 'backward':
            self.attribute()
 
        if not os.path.exists(folder):
            os.makedirs(folder)
 
        depth = len(self.width) - 1
        thumbnails = {}
 
        # ------------------------------------------------------------------
        # Phase 1: Rendering active edges
        # ------------------------------------------------------------------
        w_large = 4.0 * scale
        fig_edge, ax_edge = plt.subplots(figsize=(w_large, w_large))
 
        for l in range(depth):
            acts_l = self.acts[l].cpu().detach().numpy()
            spline_postacts_l = self.spline_postacts[l].cpu().detach().numpy()
 
            for i in range(self.width_in[l]):
                rank = np.argsort(acts_l[:, i])
                x_vals = acts_l[:, i][rank]
 
                for j in range(self.width_out[l + 1]):
                    symbolic_mask = self.symbolic_fun[l].mask[j][i].item()
                    numeric_mask = self.act_fun[l].mask[i][j].item()
 
                    # Short-circuit: pruned edge, nothing is rendered
                    if symbolic_mask == 0. and numeric_mask == 0.:
                        continue
 
                    if symbolic_mask > 0. and numeric_mask > 0.:
                        color = 'purple'
                    elif symbolic_mask > 0. and numeric_mask == 0.:
                        color = "red"
                    else:
                        color = "black"
 
                    ax_edge.clear() # reutilizes the same Figure/Axes, does not create a new one
 
                    ax_edge.spines['top'].set_visible(False)
                    ax_edge.spines['right'].set_visible(False)
                    ax_edge.spines['bottom'].set_linewidth(0.8)
                    ax_edge.spines['left'].set_linewidth(0.8)


                    x_min, x_max, y_min, y_max = self.get_range(l, i, j, verbose=False)
                    if tick:
                        ax_edge.grid(True, linestyle=':', alpha=0.3, color='gray')
                        ax_edge.tick_params(axis="both", direction="in", pad=3, labelsize=15, colors='dimgray')
                        ax_edge.set_xticks([x_min, x_max])
                        ax_edge.set_xticklabels(['%.1f' % x_min, '%.1f' % x_max])
                        ax_edge.set_yticks([y_min, y_max])
                        ax_edge.set_yticklabels(['%.1f' % y_min, '%.1f' % y_max])
                    else:
                        ax_edge.set_xticks([])
                        ax_edge.set_yticks([])
                    
                    y_vals = spline_postacts_l[:, j, i][rank]
                    ax_edge.plot(x_vals, y_vals, color=color, lw=3)

                    if sample:
                        ax_edge.scatter(x_vals, y_vals, color=color, s=20 * scale ** 2)
                    
                    for spine in ax_edge.spines.values():
                        spine.set_color(color)
 
                    # --- A single real render: it is drawn once ---
                    fig_edge.savefig(f'{folder}/sp_{l}_{i}_{j}.png', bbox_inches="tight", dpi=edge_dpi)
 
                    # Thumbnail derived from the same already rasterized buffer (no second savefig)
                    fig_edge.canvas.draw()
                    rgba = np.asarray(fig_edge.canvas.buffer_rgba())  # HxWx4 uint8, already rendered
                    step = max(1, int(round(edge_dpi / thumb_dpi)))
                    thumbnails[(l, i, j)] = rgba[::step, ::step, :].copy()  # uint8, lightweight
 
        plt.close(fig_edge)
        gc.collect()
 
        # ------------------------------------------------------------------
        # Phase 2: Construction of the main graph and connections
        # ------------------------------------------------------------------
        def score2alpha(score):
            return np.tanh(beta * score)
 
        if metric == 'forward_n':
            scores = self.acts_scale
        elif metric == 'forward_u':
            scores = self.edge_actscale
        elif metric == 'backward':
            scores = self.edge_scores
        else:
            raise Exception(f'metric = \'{metric}\' not recognized')
 
        alpha = [score2alpha(score.cpu().detach().numpy()) for score in scores]
 
        width = np.array(self.width)
        width_in = np.array(self.width_in)
        width_out = np.array(self.width_out)
        A = 1
        y0 = 0.3
        z0 = 0.1
 
        neuron_depth = len(width)
        min_spacing = A / np.maximum(np.max(width_out), 5)
        max_neuron = np.max(width_out)
        max_num_weights = np.max(width_in[:-1] * width_out[1:])
        y1 = 0.4 / np.maximum(max_num_weights, 5)
        y2 = 0.15 / np.maximum(max_neuron, 5)
 
        fig, ax = plt.subplots(figsize=(10 * scale, 10 * scale * (neuron_depth - 1) * (y0 + z0)))
 
        DC_to_FC = ax.transData.transform
        FC_to_NFC = fig.transFigure.inverted().transform
        DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))
 
        for l in range(neuron_depth):
            n = width_in[l]
            for i in range(n):
                plt.scatter(1 / (2 * n) + i / n, l * (y0 + z0), s=min_spacing ** 2 * 10000 * scale ** 2, color='black')
 
            for i in range(n):
                if l < neuron_depth - 1:
                    n_next = width_out[l + 1]
                    N = n * n_next
                    for j in range(n_next):
                        id_ = i * n_next + j
                        symbol_mask = self.symbolic_fun[l].mask[j][i].item()
                        numerical_mask = self.act_fun[l].mask[i][j].item()
 
                        if symbol_mask == 0. and numerical_mask == 0.:
                            continue
 
                        if symbol_mask == 1. and numerical_mask > 0.:
                            color, alpha_mask = 'purple', 1.
                        elif symbol_mask == 1. and numerical_mask == 0.:
                            color, alpha_mask = "red", 1.
                        else:  # symbol_mask == 0. and numerical_mask == 1.
                            color, alpha_mask = "black", 1.
 
                        plt.plot([1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N], [l * (y0 + z0), l * (y0 + z0) + y0 / 2 - y1], color=color, lw=2 * scale, alpha=alpha[l][j][i] * alpha_mask)
                        plt.plot([1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next], [l * (y0 + z0) + y0 / 2 + y1, l * (y0 + z0) + y0], color=color, lw=2 * scale, alpha=alpha[l][j][i] * alpha_mask)
 
            if l < neuron_depth - 1:
                n_in = width_out[l + 1]
                n_out = width_in[l + 1]
                mult_id = 0
                for i in range(n_in):
                    if i < width[l + 1][0]:
                        j = i
                    else:
                        if i == width[l + 1][0]:
                            ma = self.mult_arity if isinstance(self.mult_arity, int) else self.mult_arity[l + 1][mult_id]
                            current_mult_arity = ma
                        if current_mult_arity == 0:
                            mult_id += 1
                            ma = self.mult_arity if isinstance(self.mult_arity, int) else self.mult_arity[l + 1][mult_id]
                            current_mult_arity = ma
                        j = width[l + 1][0] + mult_id
                        current_mult_arity -= 1
                    plt.plot([1 / (2 * n_in) + i / n_in, 1 / (2 * n_out) + j / n_out], [l * (y0 + z0) + y0, (l + 1) * (y0 + z0)], color='black', lw=2 * scale)
 
            plt.xlim(0, 1)
            plt.ylim(-0.1 * (y0 + z0), (neuron_depth - 1 + 0.1) * (y0 + z0))
 
        plt.axis('off')
 
        # ------------------------------------------------------------------
        # Phase 3: Insertion of thumbnails and icons (100% from RAM)
        # ------------------------------------------------------------------
        for l in range(neuron_depth - 1):
            n = width_in[l]
            for i in range(n):
                n_next = width_out[l + 1]
                N = n * n_next
                for j in range(n_next):
                    if (l, i, j) not in thumbnails:
                        continue
 
                    id_ = i * n_next + j
                    im = thumbnails[(l, i, j)]
 
                    left = DC_to_NFC([1 / (2 * N) + id_ / N - y1, 0])[0]
                    right = DC_to_NFC([1 / (2 * N) + id_ / N + y1, 0])[0]
                    bottom = DC_to_NFC([0, l * (y0 + z0) + y0 / 2 - y1])[1]
                    up = DC_to_NFC([0, l * (y0 + z0) + y0 / 2 + y1])[1]
 
                    newax = fig.add_axes([left, bottom, right - left, up - bottom])
                    newax.imshow(im, alpha=alpha[l][j][i])
                    newax.axis('off')
 
            N = n = width_out[l + 1]
            for j in range(n):
                id_ = j
                path = os.path.join(kan_dir, "assets", "img", "sum_symbol.png")
                if os.path.exists(path):
                    im = plt.imread(path)
                    left = DC_to_NFC([1 / (2 * N) + id_ / N - y2, 0])[0]
                    right = DC_to_NFC([1 / (2 * N) + id_ / N + y2, 0])[0]
                    bottom = DC_to_NFC([0, l * (y0 + z0) + y0 - y2])[1]
                    up = DC_to_NFC([0, l * (y0 + z0) + y0 + y2])[1]
                    newax = fig.add_axes([left, bottom, right - left, up - bottom])
                    newax.imshow(im)
                    newax.axis('off')
 
            N = n = width_in[l + 1]
            n_sum = width[l + 1][0]
            n_mult = width[l + 1][1]
            for j in range(n_mult):
                id_ = j + n_sum
                path = os.path.join(kan_dir, "assets", "img", "mult_symbol.png")
                if os.path.exists(path):
                    im = plt.imread(path)
                    left = DC_to_NFC([1 / (2 * N) + id_ / N - y2, 0])[0]
                    right = DC_to_NFC([1 / (2 * N) + id_ / N + y2, 0])[0]
                    bottom = DC_to_NFC([0, (l + 1) * (y0 + z0) - y2])[1]
                    up = DC_to_NFC([0, (l + 1) * (y0 + z0) + y2])[1]
                    newax = fig.add_axes([left, bottom, right - left, up - bottom])
                    newax.imshow(im)
                    newax.axis('off')
 
        if in_vars is not None:
            n = self.width_in[0]
            for i in range(n):
                text_var = f'${sympy.latex(in_vars[i])}$' if isinstance(in_vars[i], sympy.Expr) else in_vars[i]
                plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), -0.1, text_var, fontsize=40 * scale * varscale, horizontalalignment='center', verticalalignment='center')
 
        if out_vars is not None:
            n = self.width_in[-1]
            for i in range(n):
                text_var = f'${sympy.latex(out_vars[i])}$' if isinstance(out_vars[i], sympy.Expr) else out_vars[i]
                plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), (y0 + z0) * (len(self.width) - 1) + 0.15, text_var, fontsize=40 * scale * varscale, horizontalalignment='center', verticalalignment='center')
 
        if title is not None:
            plt.gcf().get_axes()[0].text(0.5, (y0 + z0) * (len(self.width) - 1) + 0.3, title, fontsize=40 * scale, horizontalalignment='center', verticalalignment='center')
 
        if save_path:
            print(f'saving figure to {save_path}')
            plt.savefig(save_path, bbox_inches="tight", dpi=400)
            plt.close(fig)
        else:
            plt.show()
 
        del thumbnails
        gc.collect()

    def symbolic_formula(self, var=None, normalizer=None, output_normalizer = None):
        '''
        get symbolic formula

        Args:
            - var: None or a list of sympy expression input variables
            - normalizer: [mean, std]
            - output_normalizer: [mean, std]
            
        Returns:
        --------
            None

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,1,1], grid=5, k=3, noise_scale=0.0, seed=0)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]])+x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=3)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.auto_symbolic()
        >>> model.symbolic_formula()[0][0]
        '''
        
        symbolic_acts = []
        symbolic_acts_premult = []
        x = []

        def ex_round(ex1, n_digit):
            ex2 = ex1
            for a in sympy.preorder_traversal(ex1):
                if isinstance(a, sympy.Float):
                    ex2 = ex2.subs(a, round(a, n_digit))
            return ex2

        # define variables
        if var == None:
            for ii in range(1, self.width[0][0] + 1):
                exec(f"x{ii} = sympy.Symbol('x_{ii}')")
                exec(f"x.append(x{ii})")
        elif isinstance(var[0], sympy.Expr):
            x = var
        else:
            x = [sympy.symbols(var_) for var_ in var]

        x0 = x

        if normalizer != None:
            mean = normalizer[0]
            std = normalizer[1]
            x = [(x[i] - mean[i]) / std[i] for i in range(len(x))]

        symbolic_acts.append(x)

        for l in range(len(self.width_in) - 1):
            num_sum = self.width[l + 1][0]
            num_mult = self.width[l + 1][1]
            y = []
            for j in range(self.width_out[l + 1]):
                yj = 0.
                for i in range(self.width_in[l]):
                    affine_params = self.symbolic_fun[l].affine[j, i]#.detach().cpu().numpy()
                    a = affine_params[0].item()
                    b = affine_params[1].item()
                    c = affine_params[2].item()
                    d = affine_params[3].item()
                    sympy_fun = self.symbolic_fun[l].funs_sympy[j][i]
                    try:
                        yj += c * sympy_fun(a * x[i] + b) + d
                    except:
                        print('make sure all activations need to be converted to symbolic formulas first!')
                        return
                yj = self.subnode_scale[l][j].item() * yj + self.subnode_bias[l][j].item()
                if simplify == True:
                    y.append(sympy.simplify(yj))
                else:
                    y.append(yj)
                    
            symbolic_acts_premult.append(y)
                  
            mult = []
            for k in range(num_mult):
                if isinstance(self.mult_arity, int):
                    mult_arity = self.mult_arity
                else:
                    mult_arity = self.mult_arity[l+1][k]
                for i in range(mult_arity-1):
                    if i == 0:
                        mult_k = y[num_sum+2*k] * y[num_sum+2*k+1]
                    else:
                        mult_k = mult_k * y[num_sum+2*k+i+1]
                mult.append(mult_k)
                
            y = y[:num_sum] + mult
            
            for j in range(self.width_in[l+1]):
                y[j] = self.node_scale[l][j].item() * y[j] + self.node_bias[l][j].item()
            
            x = y
            symbolic_acts.append(x)

        if output_normalizer != None:
            output_layer = symbolic_acts[-1]
            means = output_normalizer[0]
            stds = output_normalizer[1]

            assert len(output_layer) == len(means), 'output_normalizer does not match the output layer'
            assert len(output_layer) == len(stds), 'output_normalizer does not match the output layer'
            
            output_layer = [(output_layer[i] * stds[i] + means[i]) for i in range(len(output_layer))]
            symbolic_acts[-1] = output_layer


        self.symbolic_acts = [[symbolic_acts[l][i] for i in range(len(symbolic_acts[l]))] for l in range(len(symbolic_acts))]
        self.symbolic_acts_premult = [[symbolic_acts_premult[l][i] for i in range(len(symbolic_acts_premult[l]))] for l in range(len(symbolic_acts_premult))]

        out_dim = len(symbolic_acts[-1])
        #return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0
        
        if simplify:
            return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0
        else:
            return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0

    def saveckpt(self, path='model'):
        """
        Save the model's configuration, state, and cache data to files.
        
        This method ensures that the model's base function name is stored 
        as a string, which is necessary for proper serialization and deserialization.
        """
        model = self
        
        # Forzar que el atributo interno vuelva a ser un string si es un módulo de PyTorch
        if hasattr(model, 'base_fun_name') and not isinstance(model.base_fun_name, str):
            model.base_fun_name = 'silu'# # Valor predeterminado seguro para tu arquitectura
            
        dic = dict(
            width = model.width,
            grid = model.grid,
            k = model.k,
            mult_arity = model.mult_arity,
            base_fun_name = model.base_fun_name,
            symbolic_enabled = model.symbolic_enabled,
            affine_trainable = model.affine_trainable,
            grid_eps = model.grid_eps,
            grid_range = model.grid_range,
            sp_trainable = model.sp_trainable,
            sb_trainable = model.sb_trainable,
            state_id = model.state_id,
            auto_save = model.auto_save,
            ckpt_path = model.ckpt_path,
            round = model.round,
            device = str(model.device)
        )
        
        if dic["device"].isdigit():
            dic["device"] = int(model.device)

        for i in range (model.depth):
            dic[f'symbolic.funs_name.{i}'] = model.symbolic_fun[i].funs_name

        with open(f'{path}_config.yml', 'w') as outfile:
            yaml.dump(dic, outfile, default_flow_style=False)

        torch.save(model.state_dict(), f'{path}_state')
        torch.save(model.cache_data, f'{path}_cache_data')