import numpy as np
from scipy.stats import norm
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Pricer", layout="wide")


def compute_d1(S, K, T, r, sigma, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1


def compute_d2(d1, sigma, T):
    d2 = d1 - sigma * np.sqrt(T)
    return d2


def compute_black_scholes_price(option_type, S, K, T, r, sigma, q):
    d1 = compute_d1(S, K, T, r, sigma, q)
    d2 = compute_d2(d1, sigma, T)
    
    if option_type=="Call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type=="Put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
    return price


def compute_greeks(option_type, S, K, T, r, sigma, q):
    d1 = compute_d1(S, K, T, r, sigma, q)
    d2 = compute_d2(d1, sigma, T)
    
    if option_type=="Call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = ( - (S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * norm.cdf(d2)
                  + q * S * np.exp(-q * T) * norm.cdf(d1) ) / 252
        vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1) / 100
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        
    elif option_type=="Put":
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = ( - (S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T))
                  + r * K * np.exp(-r * T) * norm.cdf(-d2)
                  - q * S * np.exp(-q * T) * norm.cdf(-d1) ) / 252
        vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1) / 100
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
    return delta, gamma, theta, vega, rho


def plot_payoff(option_type, S, K):
    S_T = np.linspace(0.5 * S, 1.5 * S, 200)  # 50% à 150% du spot

    if option_type == "Call":
        payoff = np.maximum(S_T - K, 0)
    elif option_type == "Put":
        payoff = np.maximum(K - S_T, 0)
    else:
        st.error("Type d'option invalide")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_T, y=payoff, mode='lines', name=f"{option_type} Payoff"))
    
    # Add Strike Line
    fig.add_vline(x=K, line_width=1, line_dash="dot", line_color="black", annotation_text="Strike K")
    
    # Add Spot Point
    # Calculate payoff at current Spot S
    if option_type == "Call":
        payoff_at_S = max(S - K, 0)
    else:
        payoff_at_S = max(K - S, 0)
        
    fig.add_trace(go.Scatter(x=[S], y=[payoff_at_S], mode='markers', name="Spot Price", marker=dict(color="red", size=10)))

    fig.update_layout(
        title=f"{option_type} Option Payoff at Maturity",
        xaxis_title="Price at Maturity S_T",
        yaxis_title="Payoff",
        legend_title="Legend"
    )
    return fig



def plot_all_greeks(option_type, S, K, T, r, sigma, q, S_range_factor=0.5):
    S_min = S * (1 - S_range_factor)
    S_max = S * (1 + S_range_factor)
    S_values = np.linspace(S_min, S_max, 200)

    deltas, gammas, thetas, vegas, rhos = [], [], [], [], []

    for s in S_values:
        delta, gamma, theta, vega, rho = compute_greeks(option_type, s, K, T, r, sigma, q)
        deltas.append(delta)
        gammas.append(gamma)
        thetas.append(theta)
        vegas.append(vega)
        rhos.append(rho)

    # Calculate Greeks at current Spot S
    curr_delta, curr_gamma, curr_theta, curr_vega, curr_rho = compute_greeks(option_type, S, K, T, r, sigma, q)

    # Création des figures
    delta_fig = go.Figure()
    delta_fig.add_trace(go.Scatter(x=S_values, y=deltas, mode='lines', name="Delta", line=dict(color="blue")))
    delta_fig.add_vline(x=K, line_width=1, line_dash="dot", line_color="black", annotation_text="Strike K")
    delta_fig.add_trace(go.Scatter(x=[S], y=[curr_delta], mode='markers', name="Current Delta", marker=dict(color="red", size=10)))
    delta_fig.update_layout(title=f"{option_type} Option Delta vs Spot Price", xaxis_title="Spot Price", yaxis_title="Delta")

    gamma_fig = go.Figure()
    gamma_fig.add_trace(go.Scatter(x=S_values, y=gammas, mode='lines', name="Gamma", line=dict(color="green")))
    gamma_fig.add_vline(x=K, line_width=1, line_dash="dot", line_color="black", annotation_text="Strike K")
    gamma_fig.add_trace(go.Scatter(x=[S], y=[curr_gamma], mode='markers', name="Current Gamma", marker=dict(color="red", size=10)))
    gamma_fig.update_layout(title=f"{option_type} Option Gamma vs Spot Price", xaxis_title="Spot Price", yaxis_title="Gamma")

    theta_fig = go.Figure()
    theta_fig.add_trace(go.Scatter(x=S_values, y=thetas, mode='lines', name="Theta", line=dict(color="orange")))
    theta_fig.add_vline(x=K, line_width=1, line_dash="dot", line_color="black", annotation_text="Strike K")
    theta_fig.add_trace(go.Scatter(x=[S], y=[curr_theta], mode='markers', name="Current Theta", marker=dict(color="red", size=10)))
    theta_fig.update_layout(title=f"{option_type} Option Theta vs Spot Price", xaxis_title="Spot Price", yaxis_title="Theta")

    vega_fig = go.Figure()
    vega_fig.add_trace(go.Scatter(x=S_values, y=vegas, mode='lines', name="Vega", line=dict(color="purple")))
    vega_fig.add_vline(x=K, line_width=1, line_dash="dot", line_color="black", annotation_text="Strike K")
    vega_fig.add_trace(go.Scatter(x=[S], y=[curr_vega], mode='markers', name="Current Vega", marker=dict(color="red", size=10)))
    vega_fig.update_layout(title=f"{option_type} Option Vega vs Spot Price", xaxis_title="Spot Price", yaxis_title="Vega")

    rho_fig = go.Figure()
    rho_fig.add_trace(go.Scatter(x=S_values, y=rhos, mode='lines', name="Rho", line=dict(color="brown")))
    rho_fig.add_vline(x=K, line_width=1, line_dash="dot", line_color="black", annotation_text="Strike K")
    rho_fig.add_trace(go.Scatter(x=[S], y=[curr_rho], mode='markers', name="Current Rho", marker=dict(color="red", size=10)))
    rho_fig.update_layout(title=f"{option_type} Option Rho vs Spot Price", xaxis_title="Spot Price", yaxis_title="Rho")

    # Retourne un dictionnaire avec toutes les figures
    return delta_fig, gamma_fig,theta_fig,vega_fig,rho_fig




col1, col2 = st.columns([1,3])

with col1:
    with st.container(border=True):
        st.markdown("#### Parameters")
        option_type= st.pills("Option type", ["Call","Put"], default="Call")
        S = st.number_input("Spot price", value=100, min_value=0)
        K = st.number_input("Strike", value=100, min_value=0)
        T = st.number_input("Time to maturity (in years)", value=1.0, min_value=0.001)
        r = st.number_input("Risk-free rate", value=0.05)
        sigma = st.number_input("Volatility", value=0.2, min_value=0.001)
        q = st.number_input("Dividend yield", value=0.02)

price = compute_black_scholes_price(option_type, S, K, T, r, sigma, q)
delta, gamma, theta, vega, rho = compute_greeks(option_type, S, K, T, r, sigma, q)
fig = plot_payoff(option_type, S, K)
delta_fig, gamma_fig,theta_fig,vega_fig,rho_fig = plot_all_greeks(option_type, S, K, T, r, sigma, q)

with col2:
    
    col1,col2,col3,col4,col5,col6 = st.columns(6)
    with col1:
        with st.container(border=True):
            st.metric(label="Price", value=round(price,2))
    with col2:
        with st.container(border=True):
            st.metric(label="Delta", value=round(delta,2))
    with col3:
        with st.container(border=True):
            st.metric(label="Gamma", value=round(gamma,2))
    with col4:
        with st.container(border=True):
            st.metric(label="Vega", value=round(vega,2))
    with col5:
        with st.container(border=True):
            st.metric(label="Theta", value=round(theta,2))
    with col6:
        with st.container(border=True):
            st.metric(label="Rho", value=round(rho,2))
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Payoff","Delta", "Gamma", "Vega", "Theta", "Rho"])
    
    with tab1:
        st.plotly_chart(fig, use_container_width=True)       
    with tab2:
        st.plotly_chart(delta_fig, use_container_width=True) 
    with tab3:
        st.plotly_chart(gamma_fig, use_container_width=True)    
    with tab4:
        st.plotly_chart(vega_fig, use_container_width=True)   
    with tab5:
        st.plotly_chart(theta_fig, use_container_width=True)   
    with tab6:
        st.plotly_chart(rho_fig, use_container_width=True)


