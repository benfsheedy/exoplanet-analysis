import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


# create page
st.set_page_config(
    page_title="Exoplanet Discovery Trends",
    page_icon="🪐",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
    section[data-testid="stSidebar"] h1 { margin-bottom: 0.5rem; }
    h1 { margin-bottom: 0rem; }
    h2 { margin-top: 0rem; }
</style>
""", unsafe_allow_html=True)


# constants
MISSION_EVENTS = [
    (2009, 'Kepler launch'),
    (2018, 'TESS launch'),
    (2022, 'JWST operations'),
]

FACILITY_MAP = {
    'Kepler': 'Kepler',
    'K2': 'Kepler',
    'Transiting Exoplanet Survey Satellite (TESS)': 'TESS',
}

BASE_COLORS = {
    'Observatories': '#a8d9a8',
    'Kepler': '#3182bd',
    'TESS': '#9e9ac8',
}


# data loading
@st.cache_data
def load_data():
    cols = "pl_name,disc_year,discoverymethod,disc_facility,sy_dist,pl_rade,rowupdate"
    url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+{cols}+from+ps+where+disc_year+is+not+null&format=csv"
    df = pd.read_csv(url, low_memory=False)
    df['sy_dist'] = df['sy_dist'] * 3.26156
    df = (df.sort_values('rowupdate', ascending=False)
            .drop_duplicates(subset='pl_name')
            .reset_index(drop=True))
    return df

df = load_data()


# sidebar controls
with st.sidebar:
    st.title("Exoplanet Discovery Trends")
    st.markdown("---")
    st.markdown("**Controls**")

    year_range = st.slider(
        "Discovery year range",
        min_value=2006,
        max_value=2026,
        value=(2006, 2026),
        step=1,
    )

    selected_mission = st.selectbox(
        "Filter mean charts by mission",
        options=["All", "Kepler", "TESS", "Ground Observatories"],
        index=0,
    )

    show_mission_lines = st.checkbox("Show mission launch lines", value=True)
    show_trendlines = st.checkbox("Show trendlines", value=True)

year_lo, year_hi = year_range

# df_filtered only applies to charts 2 and 3
if selected_mission == "Kepler":
    df_filtered = df[df['disc_facility'].isin(['Kepler', 'K2'])].copy()
elif selected_mission == "TESS":
    df_filtered = df[df['disc_facility'] == 'Transiting Exoplanet Survey Satellite (TESS)'].copy()
elif selected_mission == "Ground Observatories":
    df_filtered = df[~df['disc_facility'].isin(['Kepler', 'K2', 'Transiting Exoplanet Survey Satellite (TESS)'])].copy()
else:
    df_filtered = df.copy()


# ── Chart 1: Discoveries per year ────────────────────────────────────────────
st.subheader("Exoplanet Discoveries Per Year")

df_plot_filtered = df[(df['disc_year'] >= year_lo) & (df['disc_year'] <= year_hi)].copy()
df_plot_filtered['facility_group'] = (
    df_plot_filtered['disc_facility'].map(FACILITY_MAP).fillna('Observatories')
)

facility_year = df_plot_filtered.groupby(['disc_year', 'facility_group'])['pl_name'].count().unstack(fill_value=0)

cols_to_show = ['Observatories', 'Kepler', 'TESS']
for col in cols_to_show:
    if col not in facility_year.columns:
        facility_year[col] = 0
facility_year = facility_year[cols_to_show]
colors = [BASE_COLORS[c] for c in cols_to_show]

fig1, ax1 = plt.subplots(figsize=(14, 5))
facility_year.plot(kind='bar', stacked=True, ax=ax1, color=colors, width=0.85)

label_threshold = 40
for bar in ax1.patches:
    height = bar.get_height()
    if height >= label_threshold:
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_y() + height / 2,
                 int(height),
                 ha='center', va='center',
                 color='white', fontweight='bold', fontsize=7)

totals = facility_year.sum(axis=1)
ax1.bar(range(len(totals)), totals, width=0.85,
        facecolor='none', edgecolor='black', linewidth=0.3, zorder=5)

if show_mission_lines:
    ypos = ax1.get_ylim()[1] * 0.95
    for year, label in MISSION_EVENTS:
        if year_lo <= year <= year_hi and year in facility_year.index:
            x = facility_year.index.get_loc(year) - 0.43
            ax1.axvline(x=x, alpha=0.3, color='#222222', linestyle='-', linewidth=1)
            ax1.text(x + 0.1, ypos, label, fontsize=8, va='top', color='#222222', fontstyle='italic')

ax1.set_xlabel('')
ax1.set_ylabel('Exoplanets Discovered', fontsize=11)
ax1.set_xticklabels(facility_year.index.astype(int), rotation=45, ha='right', fontsize=8)
ax1.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderpad=0.4, handletextpad=0.5, fontsize=9)
ax1.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5, zorder=0)

sns.despine()
plt.tight_layout()
st.pyplot(fig1)
st.caption("Kepler's batch data releases in 2014 and 2016 confirmed nearly 2,250 planets in two years. Since 2022, TESS has taken over as the primary discovery engine.")


# ── Charts 2 & 3: side by side ───────────────────────────────────────────────
col_left, col_right = st.columns(2)

# mean distance
with col_left:
    st.subheader("Mean Distance of Discovered Exoplanets")

    df_year = (
        df_filtered.groupby('disc_year')
          .agg(mean_dist=('sy_dist', 'mean'))
          .reset_index()
          .dropna(subset=['mean_dist'])
          .query(f'{year_lo} <= disc_year <= {year_hi}')
    )

    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.plot(df_year['disc_year'], df_year['mean_dist'],
             color='#264653', linewidth=1.5, zorder=2)
    ax2.scatter(df_year['disc_year'], df_year['mean_dist'],
                color='#264653', s=30, zorder=3,
                edgecolor='white', linewidth=1)

    if show_trendlines and not df_year.empty:
        z2 = np.polyfit(df_year['disc_year'], df_year['mean_dist'], 1)
        p2 = np.poly1d(z2)
        ax2.plot(df_year['disc_year'], p2(df_year['disc_year']),
                 color='#e07b54', linewidth=1.5, linestyle='--', alpha=0.7)
        ax2.legend(fontsize=8)

    if not df_year.empty:
        ymax = df_year['mean_dist'].max() * 1.09
        ax2.set_ylim(top=ymax)
        ax2.set_xlim(df_year['disc_year'].min() - 0.5, df_year['disc_year'].max() + 0.5)

    if show_mission_lines:
        for year, label in MISSION_EVENTS:
            if not df_year.empty and df_year['disc_year'].min() <= year <= df_year['disc_year'].max():
                ax2.axvline(x=year, alpha=0.3, color='#222222', linewidth=1)
                ax2.text(year + 0.1, 0.98, label, fontsize=7, fontstyle='italic', va='top',
                         color='#222222', transform=ax2.get_xaxis_transform())

    ax2.set_ylabel('Mean Distance (light years)', fontsize=10)
    ax2.set_xticks(df_year['disc_year'])
    ax2.set_xticklabels(df_year['disc_year'].astype(int), rotation=45, ha='right', fontsize=7)
    ax2.tick_params(axis='x', labelsize=7)
    ax2.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5, zorder=0)

    sns.despine()
    plt.tight_layout()
    st.pyplot(fig2)
    st.caption("Kepler pushed mean discovery distance from under 1,000 to over 3,000 light-years by 2022. TESS, designed for nearby bright stars, has since pulled that average back down.")


# mean radius
with col_right:
    st.subheader("Mean Radius of Discovered Exoplanets")

    radius_year = (
        df_filtered[df_filtered['pl_rade'] < 30]
          .groupby('disc_year')['pl_rade'].mean()
          .reset_index()
          .query(f'{year_lo} <= disc_year <= {year_hi}')
    )

    fig3, ax3 = plt.subplots(figsize=(7, 3.5))
    ax3.plot(radius_year['disc_year'], radius_year['pl_rade'],
             color='#264653', linewidth=1.5, zorder=2)
    ax3.scatter(radius_year['disc_year'], radius_year['pl_rade'],
                color='#264653', s=30, zorder=3,
                edgecolor='white', linewidth=1)

    if show_trendlines and not radius_year.empty:
        z3 = np.polyfit(radius_year['disc_year'], radius_year['pl_rade'], 1)
        p3 = np.poly1d(z3)
        ax3.plot(radius_year['disc_year'], p3(radius_year['disc_year']),
                 color='#e07b54', linewidth=1.5, linestyle='--', alpha=0.7)
        ax3.legend(fontsize=8)

    if not radius_year.empty:
        ymax = radius_year['pl_rade'].max() * 1.09
        ax3.set_ylim(top=ymax)
        ax3.set_xlim(radius_year['disc_year'].min() - 0.5, radius_year['disc_year'].max() + 0.5)

    if show_mission_lines:
        for year, label in MISSION_EVENTS:
            if not radius_year.empty and radius_year['disc_year'].min() <= year <= radius_year['disc_year'].max():
                ax3.axvline(x=year, alpha=0.3, color='#222222', linewidth=1)
                ax3.text(year + 0.1, 0.98, label, fontsize=7, fontstyle='italic', va='top',
                         color='#222222', transform=ax3.get_xaxis_transform())

    ax3.set_ylabel('Mean Radius (Earth radii)', fontsize=10)
    ax3.set_xticks(radius_year['disc_year'])
    ax3.set_xticklabels(radius_year['disc_year'].astype(int), rotation=45, ha='right', fontsize=7)
    ax3.tick_params(axis='x', labelsize=7)
    ax3.grid(axis='y', alpha=0.2, linewidth=0.5, zorder=0)

    sns.despine()
    plt.tight_layout()
    st.pyplot(fig3)
    st.caption("Before Kepler, radial velocity surveys favored large gas giants. Kepler's transit method revealed that small rocky planets (1–2 Earth radii) are the most common type in the galaxy.")

