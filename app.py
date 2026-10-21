"""ML Model Dashboard: Classification, Regression and Clustering tracks.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

import utils as u

st.set_page_config(page_title="ML Model Dashboard", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      div[data-testid="stMetric"] {
          background: rgba(120,120,140,0.10);
          border: 1px solid rgba(120,120,140,0.20);
          padding: 12px 16px; border-radius: 12px;
      }
      .pred-card {
          border-radius: 16px; padding: 22px 26px; margin: 6px 0 14px 0;
          background: linear-gradient(135deg, rgba(76,120,255,0.18), rgba(160,90,255,0.18));
          border: 1px solid rgba(120,120,200,0.35);
      }
      .pred-card .small {opacity: .75; font-size: .85rem; letter-spacing: .04em; text-transform: uppercase;}
      .pred-card .big {font-size: 2.2rem; font-weight: 700; line-height: 1.25;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_clf():
    return u.load_clf_data(), u.load_clf_tables()


@st.cache_data(show_spinner=False)
def get_reg():
    return u.load_reg_data(), u.load_reg_tables()


@st.cache_data(show_spinner=False)
def get_clu():
    return u.load_clu_data(), u.load_clu_tables()


@st.cache_resource(show_spinner="Loading model…")
def get_clf_model(file_name: str):
    return u.load_clf_model(file_name)


@st.cache_resource(show_spinner="Loading model…")
def get_reg_bundle(file_name: str):
    return u.load_reg_bundle(file_name)


@st.cache_resource(show_spinner="Loading model…")
def get_clu_bundle(file_name: str):
    return u.load_clu_bundle(file_name)


def show(fig, **kwargs):
    st.plotly_chart(fig, width="stretch", **kwargs)


def tidy(fig, height=420, legend_bottom=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10))
    if legend_bottom:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0))
    return fig


# --------------------------------------------------------------------------- #
# Shared input widgets
# --------------------------------------------------------------------------- #
def feature_specs(df: pd.DataFrame, columns: list[str], binary: list[str] | None = None) -> dict:
    """Slider/checkbox specs derived from the real data's ranges."""
    binary = binary or []
    specs = {}
    for col in columns:
        if col in binary:
            specs[col] = dict(kind="bool", default=bool(round(df[col].median())))
            continue
        s = df[col]
        if pd.api.types.is_integer_dtype(s):
            specs[col] = dict(
                kind="int", min=int(s.min()), max=int(s.max()),
                default=int(round(s.median())), step=1, fmt="%d",
            )
        else:
            dec = int(s.astype(str).str.split(".").str[1].str.len().fillna(0).max())
            dec = max(1, min(dec, 5))
            step = float(10 ** -dec)
            lo, hi = round(float(s.min()), dec), round(float(s.max()), dec)
            specs[col] = dict(
                kind="float", min=lo, max=hi,
                default=float(min(max(round(float(s.median()), dec), lo), hi)),
                step=step, fmt=f"%.{dec}f",
            )
    return specs


def ensure_state(prefix: str, specs: dict):
    for name, sp in specs.items():
        key = f"{prefix}_{name}"
        if key not in st.session_state:
            st.session_state[key] = sp["default"]


def set_values(prefix: str, specs: dict, values: dict, row_key: str | None = None, row_info=None):
    """Callback: push a dict of values into the widgets' session state."""
    for name, sp in specs.items():
        v = values[name]
        st.session_state[f"{prefix}_{name}"] = (
            bool(v) if sp["kind"] == "bool" else (int(v) if sp["kind"] == "int" else float(v))
        )
    if row_key:
        st.session_state[row_key] = row_info


def render_inputs(prefix: str, specs: dict, descriptions: dict | None = None, ncols: int = 3) -> dict:
    descriptions = descriptions or {}
    values = {}
    cols = st.columns(ncols)
    for i, (name, sp) in enumerate(specs.items()):
        with cols[i % ncols]:
            key = f"{prefix}_{name}"
            help_txt = descriptions.get(name)
            if sp["kind"] == "bool":
                values[name] = int(st.checkbox(name, key=key, help=help_txt))
            else:
                values[name] = st.slider(
                    name, sp["min"], sp["max"], step=sp["step"], key=key,
                    format=sp["fmt"], help=help_txt,
                )
    return values


def preset_buttons(prefix: str, specs: dict, df: pd.DataFrame, columns: list[str],
                   target: str | None, row_key: str):
    c1, c2, c3 = st.columns([1, 1, 3])

    def _random():
        row = df.sample(1).iloc[0]
        set_values(prefix, specs, {c: row[c] for c in columns}, row_key,
                   dict(values={c: (float(row[c]) if specs[c]["kind"] == "float" else int(row[c])) for c in columns},
                        actual=(row[target] if target else None), index=int(row.name)))

    def _median():
        set_values(prefix, specs, {c: specs[c]["default"] for c in columns}, row_key, None)

    c1.button("🎲 Load a random real row", on_click=_random, key=f"{prefix}_rand", width="stretch")
    c2.button("↺ Reset to typical values", on_click=_median, key=f"{prefix}_reset", width="stretch")


def pred_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f'<div class="pred-card"><div class="small">{label}</div>'
        f'<div class="big">{value}</div><div class="small">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def loaded_row_matches(row_key: str, values: dict):
    info = st.session_state.get(row_key)
    if not info:
        return None
    ok = all(np.isclose(float(values[c]), float(v)) for c, v in info["values"].items())
    return info if ok else None


# =========================================================================== #
# OVERVIEW
# =========================================================================== #
def page_overview():
    st.title("📊 ML Model Dashboard")
    st.caption("Three tracks · 24 saved models · performance comparison and live predictions")

    _, ct = get_clf()
    _, rt = get_reg()
    _, cu = get_clu()

    clf_best = ct["comparison"].sort_values("Accuracy", ascending=False).iloc[0]
    reg_best = rt["comparison"].sort_values("R²", ascending=False).iloc[0]
    clu_man = cu["manifest"]
    clu_best = clu_man.sort_values("Silhouette (test)", ascending=False).iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("📱 Classification")
        st.caption("Mobile phone price range (4 classes) · 2,000 rows · 10 algorithms")
        st.metric("Best model (accuracy)", clf_best["Algorithm"], f'{clf_best["Accuracy"]:.2%}')
    with c2:
        st.subheader("🍷 Regression")
        st.caption("Red wine quality score · 1,599 rows · 10 algorithms")
        st.metric("Best model (R²)", reg_best["Algorithm"], f'R² {reg_best["R²"]:.3f}')
    with c3:
        st.subheader("🚗 Clustering")
        st.caption("Car evaluation segments · 1,728 rows · 4 models")
        st.metric("Best silhouette (test)", clu_best["Model"], f'{clu_best["Silhouette (test)"]:.3f}')

    st.divider()
    st.markdown(
        """
        **How to use this dashboard**

        1. Pick a track in the sidebar.
        2. **Performance** shows leaderboards, charts and tuning results for every model in that track.
        3. **Model explorer** drills into a single model (confusion matrix, residuals, cluster profiles…).
        4. **Predict / Assign** lets you type in your own data point, choose any one model and get its output.
        """
    )

    st.subheader("Top model per track at a glance")
    top = pd.DataFrame(
        [
            ["Classification", clf_best["Algorithm"], "Accuracy", clf_best["Accuracy"]],
            ["Regression", reg_best["Algorithm"], "R²", reg_best["R²"]],
            ["Clustering", clu_best["Model"], "Silhouette (test)", clu_best["Silhouette (test)"]],
        ],
        columns=["Track", "Best model", "Metric", "Score"],
    )
    st.dataframe(top, hide_index=True, width="stretch",
                 column_config={"Score": st.column_config.NumberColumn(format="%.4f")})
    st.info(
        "Clustering is unsupervised, so there is no accuracy-style score. "
        "Models are ranked on silhouette, Davies-Bouldin and Calinski-Harabasz instead."
    )


# =========================================================================== #
# CLASSIFICATION
# =========================================================================== #
def page_classification():
    df, t = get_clf()
    comp = t["comparison"].copy()
    comp["Best_Params"] = comp["Best_Params"].map(u.clean_params)
    labels = t["class_labels"]
    class_order = sorted(labels)
    class_names = [labels[i] for i in class_order]

    st.title("📱 Classification · Mobile Price Range")
    st.caption("Predict the price bracket (0 Low → 3 Very High) of a phone from its specs.")

    tab_perf, tab_explore, tab_pred, tab_data = st.tabs(
        ["📈 Performance", "🔍 Model explorer", "🎯 Predict", "🗂 Dataset"]
    )

    # ------------------------------------------------------------ performance
    with tab_perf:
        best = comp.sort_values("Accuracy", ascending=False).iloc[0]
        least_over = comp.sort_values("Overfit_Gap").iloc[0]
        fastest = comp.sort_values("Tuning_Time_s").iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🏆 Best accuracy", best["Algorithm"], f'{best["Accuracy"]:.2%}')
        k2.metric("Best ROC-AUC", comp.sort_values("ROC_AUC", ascending=False).iloc[0]["Algorithm"],
                  f'{comp["ROC_AUC"].max():.4f}')
        k3.metric("Smallest overfit gap", least_over["Algorithm"], f'{least_over["Overfit_Gap"]:+.3f}')
        k4.metric("Fastest to tune", fastest["Algorithm"], f'{fastest["Tuning_Time_s"]:.1f}s')

        st.subheader("Leaderboard")
        table = comp.drop(columns=["Best_Params"]).rename(columns={"Tuning_Time_s": "Tuning time (s)"})
        pc = lambda lbl, lo=0.0, hi=1.0: st.column_config.ProgressColumn(lbl, min_value=lo, max_value=hi, format="%.4f")
        st.dataframe(
            table, hide_index=True, width="stretch",
            column_config={
                "Accuracy": pc("Accuracy"), "Precision": pc("Precision"), "Recall": pc("Recall"),
                "F1": pc("F1"), "ROC_AUC": pc("ROC AUC"),
                "Train_Accuracy": pc("Train acc."),
                "CV_Mean_Accuracy": pc("CV mean acc."),
                "Overfit_Gap": st.column_config.NumberColumn("Overfit gap", format="%.4f",
                                                             help="Train accuracy − test accuracy"),
                "CV_SD": st.column_config.NumberColumn("CV SD", format="%.4f"),
                "Tuning time (s)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

        st.subheader("Test-set metrics by algorithm")
        metrics = st.multiselect("Metrics to show", ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"],
                                 default=["Accuracy", "Precision", "Recall", "F1"], key="clf_metric_sel")
        if metrics:
            long = comp.melt(id_vars="Algorithm", value_vars=metrics, var_name="Metric", value_name="Score")
            fig = px.bar(long, x="Algorithm", y="Score", color="Metric", barmode="group",
                         category_orders={"Algorithm": comp.sort_values("Accuracy", ascending=False)["Algorithm"].tolist()},
                         title="Test metrics (sorted by accuracy)")
            fig.update_yaxes(range=[max(0, long["Score"].min() - 0.05), 1.0])
            show(tidy(fig, 450))

        a, b = st.columns(2)
        with a:
            long = comp.melt(id_vars="Algorithm", value_vars=["Train_Accuracy", "Accuracy", "CV_Mean_Accuracy"],
                             var_name="Split", value_name="Accuracy ")
            long["Split"] = long["Split"].map({"Train_Accuracy": "Train", "Accuracy": "Test",
                                               "CV_Mean_Accuracy": "CV mean"})
            fig = px.bar(long, x="Algorithm", y="Accuracy ", color="Split", barmode="group",
                         title="Train vs Test vs Cross-validation accuracy")
            fig.update_yaxes(range=[0.6, 1.02])
            show(tidy(fig, 430))
        with b:
            g = comp.sort_values("Overfit_Gap", ascending=True)
            fig = px.bar(g, x="Overfit_Gap", y="Algorithm", orientation="h",
                         color="Overfit_Gap", color_continuous_scale="RdYlGn_r",
                         title="Overfitting gap (train − test accuracy)")
            fig.update_layout(coloraxis_showscale=False)
            show(tidy(fig, 430, legend_bottom=False))

        a, b = st.columns(2)
        with a:
            fig = go.Figure()
            fig.add_bar(x=comp["Algorithm"], y=comp["CV_Mean_Accuracy"], name="CV mean accuracy",
                        error_y=dict(type="data", array=comp["CV_SD"], visible=True))
            fig.add_scatter(x=comp["Algorithm"], y=comp["Accuracy"], mode="markers", name="Test accuracy",
                            marker=dict(size=12, symbol="diamond"))
            fig.update_layout(title="CV accuracy (± SD) vs test accuracy")
            fig.update_yaxes(range=[0.6, 1.0])
            show(tidy(fig, 430))
        with b:
            fig = px.scatter(comp, x="Tuning_Time_s", y="Accuracy", text="Algorithm", log_x=True,
                             size=np.clip(comp["Overfit_Gap"].abs(), 0.02, None) * 100 + 8,
                             color="Algorithm", title="Accuracy vs tuning cost (bubble = overfit gap)",
                             labels={"Tuning_Time_s": "Tuning time (s, log scale)"})
            fig.update_traces(textposition="top center")
            fig.update_layout(showlegend=False)
            show(tidy(fig, 430, legend_bottom=False))

        st.subheader("Radar comparison")
        pick = st.multiselect("Models to compare", comp["Algorithm"].tolist(),
                              default=comp.sort_values("Accuracy", ascending=False)["Algorithm"].head(3).tolist(),
                              key="clf_radar")
        axes = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
        fig = go.Figure()
        for name in pick:
            r = comp.loc[comp["Algorithm"] == name, axes].iloc[0].tolist()
            fig.add_trace(go.Scatterpolar(r=r + r[:1], theta=axes + axes[:1], fill="toself", name=name))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0.6, 1.0])), title="Metric profile")
        show(tidy(fig, 480))

        st.subheader("Hyperparameter tuning: default vs tuned")
        tune = t["tuning"].copy()
        tune["Best parameters"] = tune["Best parameters"].map(u.clean_params)
        long = tune.melt(id_vars="Algorithm", value_vars=["Default test acc", "Tuned test acc"],
                         var_name="Config", value_name="Test accuracy")
        fig = px.bar(long, x="Algorithm", y="Test accuracy", color="Config", barmode="group",
                     title="Effect of tuning on test accuracy")
        show(tidy(fig, 430))
        st.dataframe(
            tune.sort_values("Change", ascending=False), hide_index=True, width="stretch",
            column_config={
                "Best CV accuracy": st.column_config.NumberColumn(format="%.4f"),
                "Default test acc": st.column_config.NumberColumn(format="%.4f"),
                "Tuned test acc": st.column_config.NumberColumn(format="%.4f"),
                "Change": st.column_config.NumberColumn(format="%+.4f"),
                "Search time (s)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

        with st.expander("Best hyperparameters per model"):
            st.dataframe(comp[["Rank", "Algorithm", "Best_Params"]], hide_index=True, width="stretch")
        with st.expander("Earlier experiment (results_partA.csv)"):
            st.caption("An earlier, separate tuning run of two models. Kept for reference.")
            partA = t["part_a"].copy()
            partA["Best_Params"] = partA["Best_Params"].map(u.clean_params)
            st.dataframe(partA, hide_index=True, width="stretch")

    # --------------------------------------------------------------- explorer
    with tab_explore:
        preds = t["predictions"]
        algos = comp.sort_values("Rank")["Algorithm"].tolist()
        st.caption(f"Computed from the held-out test set saved in predictions_all_10_models.csv ({len(preds)} rows).")

        # per-class recall heatmap for all models
        rec = {}
        for a in algos:
            cm = confusion_matrix(preds["Actual"], preds[a], labels=class_order)
            rec[a] = cm.diagonal() / cm.sum(axis=1)
        rec_df = pd.DataFrame(rec, index=class_names).T
        fig = px.imshow(rec_df, text_auto=".2f", aspect="auto", color_continuous_scale="Blues",
                        zmin=0.4, zmax=1.0, title="Per-class recall, every model")
        show(tidy(fig, 480, legend_bottom=False))

        st.divider()
        sel = st.selectbox("Choose a model to inspect", algos, key="clf_explore_sel")
        y_true, y_pred = preds["Actual"], preds[sel]
        row = comp.loc[comp["Algorithm"] == sel].iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f'{row["Accuracy"]:.2%}')
        m2.metric("F1 (weighted)", f'{row["F1"]:.3f}')
        m3.metric("ROC-AUC", f'{row["ROC_AUC"]:.4f}')
        m4.metric("Misclassified", f'{int((y_true != y_pred).sum())} / {len(y_true)}')

        a, b = st.columns(2)
        with a:
            normalize = st.toggle("Normalise rows (show recall %)", value=False, key="clf_cm_norm")
            cm = confusion_matrix(y_true, y_pred, labels=class_order)
            z = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
            fig = px.imshow(z, x=class_names, y=class_names, text_auto=".0%" if normalize else True,
                            color_continuous_scale="Blues",
                            labels=dict(x="Predicted", y="Actual", color="Share" if normalize else "Count"),
                            title=f"Confusion matrix · {sel}")
            show(tidy(fig, 440, legend_bottom=False))
        with b:
            rep = pd.DataFrame(classification_report(y_true, y_pred, labels=class_order,
                                                      target_names=class_names, output_dict=True,
                                                      zero_division=0)).T
            rep = rep.drop(index=["accuracy"], errors="ignore")
            rep["support"] = rep["support"].astype(int)
            st.markdown("**Per-class report**")
            st.dataframe(rep, width="stretch", column_config={
                "precision": st.column_config.ProgressColumn("Precision", min_value=0, max_value=1, format="%.3f"),
                "recall": st.column_config.ProgressColumn("Recall", min_value=0, max_value=1, format="%.3f"),
                "f1-score": st.column_config.ProgressColumn("F1", min_value=0, max_value=1, format="%.3f"),
                "support": st.column_config.NumberColumn("Support"),
            })
            st.markdown(f"**Best parameters:** `{row['Best_Params']}`")
            st.caption("Most errors are expected between neighbouring price bands. "
                       "Check the off-diagonal cells of the matrix.")

        mrow = t["manifest"].loc[t["manifest"]["Algorithm"] == sel].iloc[0]
        model = get_clf_model(mrow["File"])
        imp = u.feature_importance(model, t["feature_columns"])
        if imp is not None:
            top = imp.head(12).iloc[::-1]
            fig = px.bar(top, x=top.values, y=top.index, orientation="h",
                         labels={"x": "Relative importance", "y": ""},
                         title=f"Top features · {sel}")
            show(tidy(fig, 420, legend_bottom=False))
        else:
            st.caption(f"{sel} does not expose built-in feature importances.")

    # ---------------------------------------------------------------- predict
    with tab_pred:
        st.subheader("Predict the price range of a phone")
        algos = comp.sort_values("Rank")["Algorithm"].tolist()
        acc_map = dict(zip(comp["Algorithm"], comp["Accuracy"]))
        sel = st.selectbox("Model", algos, key="clf_pred_model",
                           format_func=lambda a: f"{a}  ·  test acc {acc_map[a]:.1%}")
        raw_cols = [c for c in df.columns if c != u.CLF_TARGET]
        specs = feature_specs(df, raw_cols, binary=u.CLF_BINARY)
        ensure_state("clfin", specs)
        preset_buttons("clfin", specs, df, raw_cols, u.CLF_TARGET, "clf_row")

        with st.container(border=True):
            values = render_inputs("clfin", specs, u.CLF_DESCRIPTIONS, ncols=4)

        raw_row = pd.DataFrame([values])
        mrow = t["manifest"].loc[t["manifest"]["Algorithm"] == sel].iloc[0]
        model = get_clf_model(mrow["File"])
        pred, proba, classes = u.clf_predict(model, raw_row, t["feature_columns"])

        st.markdown("### Result")
        left, right = st.columns([1, 1.3])
        with left:
            sub = f"{sel} · test accuracy {acc_map[sel]:.1%}"
            if proba is not None:
                sub = f"confidence {proba[list(classes).index(pred)]:.1%} · " + sub
            pred_card("Predicted price range", labels[pred], sub)
            info = loaded_row_matches("clf_row", values)
            if info is not None:
                actual = int(info["actual"])
                if actual == pred:
                    st.success(f"✅ Matches the actual label of dataset row #{info['index']}: {labels[actual]}")
                else:
                    st.error(f"❌ Dataset row #{info['index']} is actually {labels[actual]}")
        with right:
            if proba is not None:
                pdf = pd.DataFrame({"Class": [labels[int(c)] for c in classes], "Probability": proba})
                fig = px.bar(pdf, x="Probability", y="Class", orientation="h", text_auto=".1%",
                             title="Class probabilities")
                fig.update_xaxes(range=[0, 1])
                fig.update_yaxes(categoryorder="array", categoryarray=class_names[::-1])
                show(tidy(fig, 300, legend_bottom=False))
            else:
                st.info("This model does not provide class probabilities.")

        with st.expander("Compare every model on this data point"):
            rows = []
            for _, r in t["manifest"].iterrows():
                p, pr, cl = u.clf_predict(get_clf_model(r["File"]), raw_row, t["feature_columns"])
                rows.append({"Model": r["Algorithm"], "Prediction": labels[p],
                             "Confidence": None if pr is None else float(pr[list(cl).index(p)]),
                             "Test accuracy": acc_map[r["Algorithm"]]})
            cmp_df = pd.DataFrame(rows)
            votes = cmp_df["Prediction"].value_counts()
            st.write(f"**Consensus:** {votes.index[0]} ({votes.iloc[0]} of {len(cmp_df)} models agree)")
            st.dataframe(cmp_df, hide_index=True, width="stretch", column_config={
                "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                "Test accuracy": st.column_config.NumberColumn(format="%.3f"),
            })
        st.caption("`screen_area` and `total_pixels` are engineered automatically from your screen and pixel inputs.")

    # ---------------------------------------------------------------- dataset
    with tab_data:
        st.subheader("Dataset (train.csv)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Features", len(df.columns) - 1)
        c3.metric("Classes", df[u.CLF_TARGET].nunique())
        st.dataframe(df.head(200), width="stretch")

        a, b = st.columns(2)
        with a:
            cnt = df[u.CLF_TARGET].map(labels).value_counts().reindex(class_names).reset_index()
            cnt.columns = ["Class", "Count"]
            show(tidy(px.bar(cnt, x="Class", y="Count", text="Count", title="Class balance"), 360, False))
        with b:
            feat = st.selectbox("Feature distribution by class",
                                [c for c in df.columns if c != u.CLF_TARGET], index=13, key="clf_dist_feat")
            tmp = df.assign(**{"Price range": df[u.CLF_TARGET].map(labels)})
            fig = px.histogram(tmp, x=feat, color="Price range", barmode="overlay", opacity=0.6,
                               category_orders={"Price range": class_names}, title=f"{feat} by price range")
            show(tidy(fig, 360))

        corr = df.corr(numeric_only=True)[u.CLF_TARGET].drop(u.CLF_TARGET).sort_values()
        fig = px.bar(corr, x=corr.values, y=corr.index, orientation="h",
                     labels={"x": "Correlation with price_range", "y": ""},
                     title="Correlation of each feature with the target")
        show(tidy(fig, 520, False))


# =========================================================================== #
# REGRESSION
# =========================================================================== #
def page_regression():
    df, t = get_reg()
    comp = t["comparison"].copy()
    preds = t["predictions"]
    algos = comp.sort_values("Rank by R²")["Algorithm"].tolist()
    r2_map = dict(zip(comp["Algorithm"], comp["R²"]))

    st.title("🍷 Regression · Red Wine Quality")
    st.caption("Predict the sensory quality score (3–8) of a red wine from its physicochemical properties.")

    tab_perf, tab_explore, tab_pred, tab_data = st.tabs(
        ["📈 Performance", "🔍 Model explorer", "🎯 Predict", "🗂 Dataset"]
    )

    # ------------------------------------------------------------ performance
    with tab_perf:
        best = comp.sort_values("R²", ascending=False).iloc[0]
        low_rmse = comp.sort_values("RMSE").iloc[0]
        low_mae = comp.sort_values("MAE").iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🏆 Best R²", best["Algorithm"], f'{best["R²"]:.3f}')
        k2.metric("Lowest RMSE", low_rmse["Algorithm"], f'{low_rmse["RMSE"]:.3f}')
        k3.metric("Lowest MAE", low_mae["Algorithm"], f'{low_mae["MAE"]:.3f}')
        k4.metric("Best CV R²", comp.sort_values("CV Mean R²", ascending=False).iloc[0]["Algorithm"],
                  f'{comp["CV Mean R²"].max():.3f}')

        st.info(
            "All models explain roughly 25–41% of the variance in wine quality. "
            "Quality is a subjective score and only loosely determined by chemistry, "
            "so differences between the top models are small."
        )

        st.subheader("Leaderboard")
        st.dataframe(
            comp, hide_index=True, width="stretch",
            column_config={
                "R²": st.column_config.ProgressColumn("Test R²", min_value=0, max_value=0.5, format="%.4f"),
                "Train R²": st.column_config.ProgressColumn("Train R²", min_value=0, max_value=1, format="%.4f"),
                "CV Mean R²": st.column_config.ProgressColumn("CV mean R²", min_value=0, max_value=0.5, format="%.4f"),
                "RMSE": st.column_config.NumberColumn(format="%.4f", help="Lower is better"),
                "MAE": st.column_config.NumberColumn(format="%.4f", help="Lower is better"),
                "CV SD": st.column_config.NumberColumn(format="%.4f"),
            },
        )

        a, b = st.columns(2)
        with a:
            g = comp.sort_values("R²")
            fig = px.bar(g, x="R²", y="Algorithm", orientation="h", text_auto=".3f",
                         color="R²", color_continuous_scale="Viridis", title="Test R² (higher is better)")
            fig.update_layout(coloraxis_showscale=False)
            show(tidy(fig, 450, False))
        with b:
            long = comp.melt(id_vars="Algorithm", value_vars=["RMSE", "MAE"], var_name="Error", value_name="Value")
            fig = px.bar(long, x="Algorithm", y="Value", color="Error", barmode="group",
                         category_orders={"Algorithm": algos}, title="Error metrics (lower is better)")
            fig.update_yaxes(range=[0.4, 0.75])
            show(tidy(fig, 450))

        a, b = st.columns(2)
        with a:
            long = comp.melt(id_vars="Algorithm", value_vars=["Train R²", "R²", "CV Mean R²"],
                             var_name="Split", value_name="R² ")
            long["Split"] = long["Split"].map({"Train R²": "Train", "R²": "Test", "CV Mean R²": "CV mean"})
            fig = px.bar(long, x="Algorithm", y="R² ", color="Split", barmode="group",
                         category_orders={"Algorithm": algos}, title="Train vs Test vs CV R²")
            show(tidy(fig, 450))
        with b:
            fig = go.Figure()
            fig.add_bar(x=comp["Algorithm"], y=comp["CV Mean R²"], name="CV mean R²",
                        error_y=dict(type="data", array=comp["CV SD"], visible=True))
            fig.add_scatter(x=comp["Algorithm"], y=comp["R²"], mode="markers", name="Test R²",
                            marker=dict(size=12, symbol="diamond"))
            fig.update_layout(title="Cross-validated R² (± SD) vs test R²")
            show(tidy(fig, 450))

        gap = comp.assign(**{"Overfit gap (train − test R²)": comp["Train R²"] - comp["R²"]}) \
            .sort_values("Overfit gap (train − test R²)", ascending=False)
        fig = px.bar(gap, x="Algorithm", y="Overfit gap (train − test R²)",
                     color="Overfit gap (train − test R²)", color_continuous_scale="RdYlGn_r",
                     title="Overfitting: KNN and tree-based models memorise the training data")
        fig.update_layout(coloraxis_showscale=False)
        show(tidy(fig, 380, False))

        st.subheader("Hyperparameter tuning: default vs tuned")
        tune = t["tuning"].copy()
        tune["Best parameters"] = tune["Best parameters"].map(u.clean_params)
        long = tune.melt(id_vars="Model", value_vars=["Default Test R²", "Tuned Test R²"],
                         var_name="Config", value_name="Test R²")
        fig = px.bar(long, x="Model", y="Test R²", color="Config", barmode="group",
                     text_auto=".3f", title="Tuning impact (3 tuned models)")
        show(tidy(fig, 380))
        st.dataframe(tune, hide_index=True, width="stretch", column_config={
            "Default Test R²": st.column_config.NumberColumn(format="%.4f"),
            "Tuned Test R²": st.column_config.NumberColumn(format="%.4f"),
            "Improvement": st.column_config.NumberColumn(format="%+.4f"),
        })

    # --------------------------------------------------------------- explorer
    with tab_explore:
        st.caption(f"Computed from the held-out test set in predictions_all_10_models.csv ({len(preds)} rows).")

        resid = pd.DataFrame({a: preds[a] - preds["Actual"] for a in algos})
        long = resid.melt(var_name="Algorithm", value_name="Residual")
        fig = px.box(long, x="Algorithm", y="Residual", category_orders={"Algorithm": algos},
                     title="Residual distribution, every model (closer to 0 and narrower is better)")
        fig.add_hline(y=0, line_dash="dash")
        show(tidy(fig, 430, False))

        tol = pd.DataFrame({
            "Algorithm": algos,
            "Within ±0.5": [float((resid[a].abs() <= 0.5).mean()) for a in algos],
            "Within ±1.0": [float((resid[a].abs() <= 1.0).mean()) for a in algos],
            "Exact (rounded)": [float((preds[a].round() == preds["Actual"]).mean()) for a in algos],
        })
        fig = px.bar(tol.melt(id_vars="Algorithm", var_name="Tolerance", value_name="Share of wines"),
                     x="Algorithm", y="Share of wines", color="Tolerance", barmode="group",
                     title="How often is each model close enough?")
        show(tidy(fig, 420))

        st.divider()
        sel = st.selectbox("Choose a model to inspect", algos, key="reg_explore_sel")
        y, yhat = preds["Actual"], preds[sel]
        err = yhat - y
        row = comp.loc[comp["Algorithm"] == sel].iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test R²", f'{row["R²"]:.3f}')
        m2.metric("RMSE", f'{row["RMSE"]:.3f}')
        m3.metric("MAE", f'{row["MAE"]:.3f}')
        m4.metric("Within ±1 quality point", f'{(err.abs() <= 1).mean():.1%}')

        a, b = st.columns(2)
        with a:
            rng = np.random.default_rng(0)
            sc = pd.DataFrame({"Actual": y + rng.uniform(-0.12, 0.12, len(y)), "Predicted": yhat})
            fig = px.scatter(sc, x="Actual", y="Predicted", opacity=0.55,
                             title=f"Predicted vs actual · {sel} (actual jittered for visibility)")
            lo, hi = 2.5, 8.5
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash"))
            show(tidy(fig, 430, False))
        with b:
            fig = px.histogram(err.to_frame("Residual"), x="Residual", nbins=30,
                               title=f"Residuals · {sel}")
            fig.add_vline(x=0, line_dash="dash")
            show(tidy(fig, 430, False))

        by_q = pd.DataFrame({"Actual quality": y.astype(int).astype(str), "Residual": err})
        fig = px.box(by_q, x="Actual quality", y="Residual",
                     category_orders={"Actual quality": sorted(by_q["Actual quality"].unique())},
                     title="Residual by true quality: models regress toward the middle of the scale")
        fig.add_hline(y=0, line_dash="dash")
        show(tidy(fig, 380, False))

        mrow = t["manifest"].loc[t["manifest"]["Algorithm"] == sel].iloc[0]
        bundle = get_reg_bundle(mrow["File"])
        imp = u.feature_importance(bundle["model"], bundle["feature_columns"])
        if imp is not None:
            top = imp.iloc[::-1]
            fig = px.bar(top, x=top.values, y=top.index, orientation="h",
                         labels={"x": "Relative importance (|coefficient| for linear models)", "y": ""},
                         title=f"Feature importance · {sel}")
            show(tidy(fig, 420, False))
        else:
            st.caption(f"{sel} does not expose built-in feature importances.")

    # ---------------------------------------------------------------- predict
    with tab_pred:
        st.subheader("Predict the quality of a wine")
        sel = st.selectbox("Model", algos, key="reg_pred_model",
                           format_func=lambda a: f"{a}  ·  test R² {r2_map[a]:.3f}")
        raw_cols = t["raw_columns"]
        specs = feature_specs(df, raw_cols)
        ensure_state("regin", specs)
        preset_buttons("regin", specs, df, raw_cols, u.REG_TARGET, "reg_row")

        with st.container(border=True):
            values = render_inputs("regin", specs, ncols=4)

        raw_row = pd.DataFrame([values])
        mrow = t["manifest"].loc[t["manifest"]["Algorithm"] == sel].iloc[0]
        bundle = get_reg_bundle(mrow["File"])
        pred = float(u.reg_predict(bundle, raw_row)[0])
        rmse = float(comp.loc[comp["Algorithm"] == sel, "RMSE"].iloc[0])
        rounded = int(np.clip(round(pred), 3, 8))

        st.markdown("### Result")
        left, right = st.columns([1, 1.3])
        with left:
            tier = "Below average" if rounded <= 5 else ("Average-good" if rounded == 6 else "Good")
            pred_card("Predicted quality score", f"{pred:.2f}",
                      f"≈ {rounded}/10 · {tier} · typical error ±{rmse:.2f} ({sel})")
            info = loaded_row_matches("reg_row", values)
            if info is not None:
                actual = float(info["actual"])
                st.success(f"Dataset row #{info['index']} has actual quality **{actual:.0f}** "
                           f"(error {pred - actual:+.2f})")
        with right:
            dist = df[u.REG_TARGET].value_counts().sort_index().reset_index()
            dist.columns = ["Quality", "Wines"]
            fig = px.bar(dist, x="Quality", y="Wines", title="Where this prediction falls in the dataset")
            fig.add_vline(x=pred, line_dash="dash", line_width=3,
                          annotation_text=f"prediction {pred:.2f}", annotation_position="top")
            show(tidy(fig, 320, False))

        # input sanity check against training range
        out_of_range = [c for c in raw_cols if not (df[c].min() <= values[c] <= df[c].max())]
        if out_of_range:
            st.warning("Outside the observed range: " + ", ".join(out_of_range))

        with st.expander("Compare every model on this wine"):
            rows = []
            for _, r in t["manifest"].iterrows():
                p = float(u.reg_predict(get_reg_bundle(r["File"]), raw_row)[0])
                rows.append({"Model": r["Algorithm"], "Predicted quality": p, "Test R²": r2_map[r["Algorithm"]]})
            cmp_df = pd.DataFrame(rows).sort_values("Test R²", ascending=False)
            fig = px.bar(cmp_df, x="Model", y="Predicted quality", text_auto=".2f",
                         title="Prediction from each model")
            fig.add_hline(y=cmp_df["Predicted quality"].mean(), line_dash="dot",
                          annotation_text="mean of models")
            show(tidy(fig, 360, False))
            st.dataframe(cmp_df, hide_index=True, width="stretch",
                         column_config={"Predicted quality": st.column_config.NumberColumn(format="%.3f"),
                                        "Test R²": st.column_config.NumberColumn(format="%.3f")})
        st.caption("Engineered features, outlier clipping (IQR fences) and scaling are applied automatically, "
                   "exactly as the model saw them during training.")

    # ---------------------------------------------------------------- dataset
    with tab_data:
        st.subheader("Dataset (winequality-red.csv)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Features", len(df.columns) - 1)
        c3.metric("Quality range", f"{df[u.REG_TARGET].min()}–{df[u.REG_TARGET].max()}")
        st.dataframe(df.head(200), width="stretch")

        a, b = st.columns(2)
        with a:
            cnt = df[u.REG_TARGET].value_counts().sort_index().reset_index()
            cnt.columns = ["Quality", "Count"]
            show(tidy(px.bar(cnt, x="Quality", y="Count", text="Count", title="Target distribution"), 380, False))
        with b:
            feat = st.selectbox("Feature vs quality", [c for c in df.columns if c != u.REG_TARGET],
                                index=10, key="reg_dist_feat")
            tmp = df.assign(Quality=df[u.REG_TARGET].astype(str))
            fig = px.box(tmp, x="Quality", y=feat, category_orders={"Quality": sorted(tmp["Quality"].unique())},
                         title=f"{feat} by quality")
            show(tidy(fig, 380, False))

        corr = df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        aspect="auto", title="Correlation matrix")
        show(tidy(fig, 620, False))


# =========================================================================== #
# CLUSTERING
# =========================================================================== #
@st.cache_data(show_spinner=False)
def cluster_frame(file_name: str):
    """Assign every car to a cluster and project to 2-D with PCA (for plots)."""
    df, _ = get_clu()
    b = get_clu_bundle(file_name)
    labels = u.clu_predict(b, df)
    Xs = u.clu_scaled_features(b, df)
    pca = PCA(n_components=2, random_state=0).fit(Xs)
    proj = pca.transform(Xs)
    cent = pca.transform(b["centroids_scaled"])
    enc = u.clu_encoded_features(b, df)
    out = pd.DataFrame({"PC1": proj[:, 0], "PC2": proj[:, 1],
                        "Cluster": ["Cluster " + str(l) for l in labels],
                        "Car class": df["class"].values})
    return dict(frame=out, centroids=cent, mean=pca.mean_, components=pca.components_,
                var=pca.explained_variance_ratio_, enc=enc, labels=labels)


def page_clustering():
    df, t = get_clu()
    man = t["manifest"]
    mapping = t["ordinal_mappings"]

    st.title("🚗 Clustering · Car Evaluation")
    st.caption("Unsupervised segmentation of cars by price, capacity and safety attributes. "
               "The car 'class' column is not used for fitting; it is only used afterwards to interpret clusters.")

    tab_perf, tab_explore, tab_pred, tab_data = st.tabs(
        ["📈 Performance", "🔍 Cluster explorer", "🎯 Assign a cluster", "🗂 Dataset"]
    )

    rows = []
    for _, r in man.iterrows():
        b = get_clu_bundle(r["File"])
        m = b["metrics"]
        rows.append({
            "Model": r["Model"], "k": int(r["k"]), "Algorithm": b["algorithm"],
            "Silhouette (train)": m["Silhouette (train)"], "Silhouette (test)": m["Silhouette (test)"],
            "Davies-Bouldin": m["Davies-Bouldin (full)"], "Calinski-Harabasz": m["Calinski-Harabasz (full)"],
            "ARI (post-hoc)": m["ARI (post-hoc)"], "NMI (post-hoc)": m["NMI (post-hoc)"],
        })
    perf = pd.DataFrame(rows)
    models = perf["Model"].tolist()
    file_map = dict(zip(man["Model"], man["File"]))

    # ------------------------------------------------------------ performance
    with tab_perf:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🏆 Best silhouette (test)", perf.sort_values("Silhouette (test)", ascending=False).iloc[0]["Model"],
                  f'{perf["Silhouette (test)"].max():.3f}')
        k2.metric("Lowest Davies-Bouldin", perf.sort_values("Davies-Bouldin").iloc[0]["Model"],
                  f'{perf["Davies-Bouldin"].min():.3f}')
        k3.metric("Highest Calinski-Harabasz", perf.sort_values("Calinski-Harabasz", ascending=False).iloc[0]["Model"],
                  f'{perf["Calinski-Harabasz"].max():.0f}')
        k4.metric("Best NMI vs car class", perf.sort_values("NMI (post-hoc)", ascending=False).iloc[0]["Model"],
                  f'{perf["NMI (post-hoc)"].max():.3f}')

        st.subheader("Model comparison")
        st.dataframe(perf, hide_index=True, width="stretch", column_config={
            "Silhouette (train)": st.column_config.ProgressColumn(min_value=0, max_value=0.5, format="%.4f"),
            "Silhouette (test)": st.column_config.ProgressColumn(min_value=0, max_value=0.5, format="%.4f"),
            "Davies-Bouldin": st.column_config.NumberColumn(format="%.3f", help="Lower is better"),
            "Calinski-Harabasz": st.column_config.NumberColumn(format="%.1f", help="Higher is better"),
            "ARI (post-hoc)": st.column_config.NumberColumn(format="%.3f"),
            "NMI (post-hoc)": st.column_config.NumberColumn(format="%.3f"),
        })
        st.warning(
            "Scores are not directly comparable across different k: silhouette and Calinski-Harabasz "
            "naturally favour fewer clusters. Compare k=2 models with each other and k=4 models with each other."
        )

        a, b = st.columns(2)
        with a:
            long = perf.melt(id_vars="Model", value_vars=["Silhouette (train)", "Silhouette (test)"],
                             var_name="Split", value_name="Silhouette")
            fig = px.bar(long, x="Model", y="Silhouette", color="Split", barmode="group", text_auto=".3f",
                         title="Silhouette score (higher is better)")
            show(tidy(fig, 420))
        with b:
            fig = px.bar(perf, x="Model", y="Davies-Bouldin", color="Model", text_auto=".2f",
                         title="Davies-Bouldin index (lower is better)")
            fig.update_layout(showlegend=False)
            show(tidy(fig, 420, False))

        a, b = st.columns(2)
        with a:
            fig = px.bar(perf, x="Model", y="Calinski-Harabasz", color="Model", text_auto=".0f",
                         title="Calinski-Harabasz index (higher is better)")
            fig.update_layout(showlegend=False)
            show(tidy(fig, 420, False))
        with b:
            long = perf.melt(id_vars="Model", value_vars=["ARI (post-hoc)", "NMI (post-hoc)"],
                             var_name="Metric", value_name="Score")
            fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group", text_auto=".3f",
                         title="Agreement with the hidden car class (external, post-hoc)")
            show(tidy(fig, 420))

        sizes = []
        for m in models:
            lab = get_clu_bundle(file_map[m])["train_labels"]
            for c, n in zip(*np.unique(lab, return_counts=True)):
                sizes.append({"Model": m, "Cluster": f"Cluster {c}", "Cars (train)": int(n)})
        fig = px.bar(pd.DataFrame(sizes), x="Model", y="Cars (train)", color="Cluster", text_auto=True,
                     title="Cluster sizes on the training split")
        show(tidy(fig, 420))

    # --------------------------------------------------------------- explorer
    with tab_explore:
        sel = st.selectbox("Choose a model", models, key="clu_explore_sel")
        info = cluster_frame(file_map[sel])
        frame = info["frame"]
        st.caption("Every car in the dataset assigned with the model's nearest-centroid rule, "
                   "projected to 2-D with PCA for display only.")
        a, b = st.columns([1.4, 1])
        with a:
            fig = px.scatter(frame, x="PC1", y="PC2", color="Cluster", opacity=0.6,
                             title=f"Clusters in PCA space · {sel} "
                                   f"({info['var'].sum():.0%} of variance shown)")
            fig.add_trace(go.Scatter(x=info["centroids"][:, 0], y=info["centroids"][:, 1], mode="markers+text",
                                     text=[f"C{i}" for i in range(len(info["centroids"]))],
                                     textposition="top center", name="Centroids",
                                     marker=dict(size=16, symbol="x", color="black", line=dict(width=2))))
            show(tidy(fig, 480))
        with b:
            cnt = frame["Cluster"].value_counts().sort_index().reset_index()
            cnt.columns = ["Cluster", "Cars"]
            fig = px.bar(cnt, x="Cluster", y="Cars", text="Cars", color="Cluster",
                         title="Cluster sizes (all cars)")
            fig.update_layout(showlegend=False)
            show(tidy(fig, 480, False))

        enc = info["enc"].copy()
        enc["Cluster"] = ["Cluster " + str(l) for l in info["labels"]]
        prof = enc.groupby("Cluster").mean()
        z = (prof - enc.drop(columns="Cluster").mean()) / enc.drop(columns="Cluster").std(ddof=0)
        fig = go.Figure(go.Heatmap(
            z=z.values, x=prof.columns.tolist(), y=prof.index.tolist(),
            text=np.round(prof.values, 2), texttemplate="%{text}", colorscale="RdBu_r", zmid=0,
            colorbar=dict(title="vs avg (σ)"),
        ))
        fig.update_layout(title="Cluster profile: mean encoded feature value (colour = deviation from overall mean)")
        show(tidy(fig, 100 + 70 * len(prof), False))

        comp = pd.crosstab(frame["Cluster"], frame["Car class"], normalize="index")
        long = comp.reset_index().melt(id_vars="Cluster", var_name="Car class", value_name="Share")
        fig = px.bar(long, x="Cluster", y="Share", color="Car class", text_auto=".0%",
                     title="Car-class mix inside each cluster (post-hoc interpretation)")
        show(tidy(fig, 400))
        st.caption("Encodings: buying/maint low=1…vhigh=4 · doors 2…5 · persons 2/4/6 · "
                   "lug_boot small=1…big=3 · safety low=1…high=3. "
                   "cost_burden = buying+maint, utility_index = persons×lug_boot, safety_capacity = safety×persons.")

    # ----------------------------------------------------------------- assign
    with tab_pred:
        st.subheader("Assign a car to a cluster")
        sel = st.selectbox("Model", models, key="clu_pred_model")
        bundle = get_clu_bundle(file_map[sel])
        order = {"buying": ["low", "med", "high", "vhigh"], "maint": ["low", "med", "high", "vhigh"],
                 "doors": ["2", "3", "4", "5more"], "persons": ["2", "4", "more"],
                 "lug_boot": ["small", "med", "big"], "safety": ["low", "med", "high"]}
        nice = {"buying": "Buying price", "maint": "Maintenance cost", "doors": "Doors",
                "persons": "Passenger capacity", "lug_boot": "Boot size", "safety": "Safety rating"}

        with st.container(border=True):
            cols = st.columns(3)
            car = {}
            for i, (feat, opts) in enumerate(order.items()):
                opts = [o for o in opts if o in mapping[feat]]
                default = opts[min(1, len(opts) - 1)]
                with cols[i % 3]:
                    car[feat] = st.selectbox(nice[feat], opts, index=opts.index(default), key=f"cluin_{feat}")
        row = pd.DataFrame([car])

        d2 = u.clu_distances(bundle, row)[0]
        cl = int(d2.argmin())
        info = cluster_frame(file_map[sel])

        st.markdown("### Result")
        left, right = st.columns([1, 1.3])
        with left:
            pred_card("Assigned cluster", f"Cluster {cl}",
                      f"{sel} · nearest centroid distance {np.sqrt(d2[cl]):.2f}")
            members = info["frame"][info["frame"]["Cluster"] == f"Cluster {cl}"]
            mix = members["Car class"].value_counts(normalize=True)
            st.markdown("**Typical car class in this cluster**")
            st.write(", ".join(f"{k} {v:.0%}" for k, v in mix.items()))
            st.caption(f"{len(members):,} of {len(info['frame']):,} cars belong to this cluster.")
        with right:
            dist = pd.DataFrame({"Cluster": [f"Cluster {i}" for i in range(len(d2))],
                                 "Distance": np.sqrt(d2)})
            fig = px.bar(dist, x="Distance", y="Cluster", orientation="h", text_auto=".2f",
                         color=(dist["Cluster"] == f"Cluster {cl}").map({True: "Assigned", False: "Other"}),
                         color_discrete_map={"Assigned": "#4C78FF", "Other": "#9AA0B4"},
                         title="Distance to each cluster centroid (smaller = closer)")
            fig.update_layout(showlegend=False)
            show(tidy(fig, 300, False))

        # where the new car sits in PCA space
        xs = u.clu_scaled_features(bundle, row)
        pt = (xs - info["mean"]) @ info["components"].T
        fig = px.scatter(info["frame"], x="PC1", y="PC2", color="Cluster", opacity=0.35,
                         title="Your car (★) among all cars")
        fig.add_trace(go.Scatter(x=[pt[0, 0]], y=[pt[0, 1]], mode="markers", name="Your car",
                                 marker=dict(size=22, symbol="star", color="gold", line=dict(width=2, color="black"))))
        show(tidy(fig, 440))

        with st.expander("How this car compares with the cluster's typical profile"):
            enc_row = u.clu_encoded_features(bundle, row).iloc[0]
            enc_all = info["enc"].copy()
            prof = enc_all[info["labels"] == cl].mean()
            cmp_df = pd.DataFrame({"Your car": enc_row, f"Cluster {cl} average": prof,
                                   "All cars average": enc_all.mean()}).round(2)
            st.dataframe(cmp_df, width="stretch")

    # ---------------------------------------------------------------- dataset
    with tab_data:
        st.subheader("Dataset (car_evaluation.csv)")
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Attributes", len(df.columns) - 1)
        st.dataframe(df.head(200), width="stretch")
        feat = st.selectbox("Attribute distribution", [c for c in df.columns], key="clu_dist_feat")
        cnt = df[feat].value_counts().reset_index()
        cnt.columns = [feat, "Count"]
        show(tidy(px.bar(cnt, x=feat, y="Count", text="Count", title=f"{feat}"), 360, False))


# =========================================================================== #
# Router
# =========================================================================== #
PAGES = {
    "🏠 Overview": page_overview,
    "📱 Classification": page_classification,
    "🍷 Regression": page_regression,
    "🚗 Clustering": page_clustering,
}

with st.sidebar:
    st.header("📊 ML Dashboard")
    choice = st.radio("Track", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Models are loaded from the `saved_models` folders next to this app. "
               "Predictions apply the same feature engineering, clipping and scaling used in training.")

PAGES[choice]()
