#!/bin/bash

FINAL_FIG_DIR="final_figures"
if [[ ! -d "$FINAL_FIG_DIR" ]]; then
    mkdir -p $FINAL_FIG_DIR
fi

# Figure 1
cp figures/suplord_v_lordfdx_intro/fdx_only/intro_fig.png $FINAL_FIG_DIR/intro_fig.png


# Figure 3
cp figures/suplord_v_lordfdx_constant/all/Power.png $FINAL_FIG_DIR/fdx_constant_Power.png
# Figure 4a
cp figures/suplord_v_lordfdx_constant/all/hypotheses_Wealth.png $FINAL_FIG_DIR/fdx_constant_wealth.png
# Figure 7 cp figures/suplord_v_lordfdx_constant/all/FDR.png $FINAL_FIG_DIR/fdx_constant_FDR.png # Figure 8
cp figures/suplord_v_lordfdx_constant/all/SupFDP.png $FINAL_FIG_DIR/fdx_constant_SupFDR.png

# Figure 5
cp figures/dynamic_v_static_constant/Power.png $FINAL_FIG_DIR/dynamic_constant_Power.png
cp figures/dynamic_v_static_constant/hypotheses_Wealth.png $FINAL_FIG_DIR/dynamic_constant_wealth.png
# Figure 9
cp figures/dynamic_v_static_constant/FDR.png $FINAL_FIG_DIR/dynamic_constant_FDR.png
# Figure 10
cp figures/dynamic_v_static_constant/SupFDP.png $FINAL_FIG_DIR/dynamic_constant_SupFDR.png


# Figure 11
cp figures/suplord_v_lordfdx_hmm/Power.png $FINAL_FIG_DIR/dynamic_constant_SupFDR.png
cp figures/suplord_v_lordfdx_hmm/FDR.png $FINAL_FIG_DIR/dynamic_constant_SupFDR.png
cp figures/suplord_v_lordfdx_hmm/SupFDP.png $FINAL_FIG_DIR/dynamic_constant_SupFDR.png

# Figure 12
cp figures/dynamic_v_static_hmm/Power.png $FINAL_FIG_DIR/dynamic_constant_SupFDR.png
cp figures/dynamic_v_static_hmm/FDR.png $FINAL_FIG_DIR/dynamic_constant_SupFDR.png
cp figures/dynamic_v_static_hmm/SupFDP.png $FINAL_FIG_DIR/dynamic_constant_SupFDR.png

# Figure 13
cp figures/dynamic_v_static_constant/alpha_comp_STEADY.png $FINAL_FIG_DIR/dynamic_alpha_STEADY.png
cp figures/dynamic_v_static_constant/alpha_comp_AGGRESSIVE.png $FINAL_FIG_DIR/dynamic_alpha_AGGRESSIVE.png

# Figure 14
cp figures/dynamic_v_static_constant/alpha_run.png $FINAL_FIG_DIR/dynamic_alpha_run.png

# Figure 15
cp figures/dynamic_v_static_constant/hypotheses_Alpha.png $FINAL_FIG_DIR/dynamic_alpha_mean.png


