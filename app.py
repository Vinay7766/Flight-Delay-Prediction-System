"""Flight Delay Predictor - Streamlit app.

Runs the Random Forest and Logistic Regression trained with Spark MLlib in the
notebook. The trained trees and weights were exported to model_export.json, so
the app needs no Spark installation.
"""
import json
import math
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️", layout="wide")
ss = st.session_state


@st.cache_data
def load():
    return json.loads((Path(__file__).parent / "model_export.json").read_text())


D = load()
AL = D["airline_labels"]
AP = {a["code"]: a for a in D["airports"]}
AN = {a["AIRLINE"]: a["AIRLINE_NAME"] for a in D["airline_stats"]}
AC = {"WN": "#f97316", "DL": "#dc2626", "AA": "#64748b", "OO": "#8b5cf6", "EV": "#14b8a6",
      "UA": "#1d4ed8", "MQ": "#92400e", "B6": "#06b6d4", "US": "#78716c", "AS": "#84cc16",
      "NK": "#facc15", "F9": "#16a34a", "HA": "#ec4899", "VX": "#a21caf"}
MN = ["January", "February", "March", "April", "May", "June", "July", "August",
      "September", "October", "November", "December"]
DN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
AVG = D["totals"]["delayed_pct"]
IN_DATA = {m["MONTH"] for m in D["by_month"]}
TEAL, AMBER = "#0f7b8a", "#d97706"
RES = {r["Model"]: r for r in D["results"]}

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@800&display=swap');
h1.hero{font-family:'Fraunces',Georgia,serif;font-weight:800;letter-spacing:-.015em;line-height:1.08;font-size:clamp(2rem,5vw,3.4rem);margin:0 0 .3rem}
.big{font-size:1.9rem;font-weight:700;margin:.2rem 0}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- model code
def run_tree(t, x):
    while isinstance(t, list):
        f, kind, v, left, right = t
        go_left = (x[f] in v) if kind else (x[f] <= v)
        t = left if go_left else right
    return t


def rf_score(x):
    return sum(run_tree(t, x) for t in D["rf"]) / len(D["rf"])


def lr_score(x):
    z = D["lr"]["intercept"] + sum(c * xi / s for c, xi, s in zip(D["lr"]["coef"], x, D["lr"]["std"]))
    return 1 / (1 + math.exp(-z))


def haversine(a, b):
    p1, p2 = math.radians(a["lat"]), math.radians(b["lat"])
    dp, dl = p2 - p1, math.radians(b["lon"] - a["lon"])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 3959 * math.asin(math.sqrt(h))


def distance(o, d):
    k = f"{o}-{d}"
    if k in D["routes"]:
        return D["routes"][k], True
    a, b = AP[o], AP[d]
    if a["lat"] is None or b["lat"] is None:
        return None, False
    return round(haversine(a, b)), False


def features(inp, dist, **over):
    i = {**inp, **over}
    o, d = AP[i["o"]], AP[i["d"]]
    x = [0.0] * 24
    x[AL.index(i["al"])] = 1.0
    x[14:24] = [i["mo"], i["dw"], i["hr"], dist, o["pr"], o["inn"], o["out"], d["pr"], d["inn"], d["out"]]
    return x


def hl(h):
    return f"{h % 12 or 12}:00 {'am' if h < 12 else 'pm'}"


def period(h):
    return "night" if h < 5 or h >= 21 else "morning" if h < 12 else "afternoon" if h < 17 else "evening"


def check(o, d, mo, hr):
    w = []
    if o == d:
        w.append("Choose two different airports.")
    elif f"{o}-{d}" not in D["routes"]:
        w.append(f"No {o} to {d} flights appear in the 2015 sample, so the distance is estimated "
                 "and the result is less reliable.")
    if mo is None:
        w.append("Pick a month.")
    elif mo not in IN_DATA:
        w.append(f"{MN[mo - 1]} has no flights in the training data, so the model is extrapolating.")
    if hr < 5:
        w.append("Few flights are scheduled at this hour, so the estimate is less reliable.")
    return w


# ---------------------------------------------------------------- state
for k, v in dict(o="ATL", d="LAX", al="DL", mo=6, dw=5, hr=18, res=None, msg="").items():
    ss.setdefault(k, v)


def swap():
    ss.o, ss.d = ss.d, ss.o


def example(p):
    ss.update(p)
    ss.msg = "Example loaded. Press Predict delay risk."


EXAMPLES = [("Evening ATL to LAX, Delta", dict(o="ATL", d="LAX", al="DL", mo=6, dw=5, hr=18)),
            ("Morning JFK to MIA, American", dict(o="JFK", d="MIA", al="AA", mo=3, dw=1, hr=8))]

# ---------------------------------------------------------------- header
st.markdown('<h1 class="hero">Will your flight arrive late?</h1>', unsafe_allow_html=True)
st.caption("The Spark MLlib models from this project score your flight. Airport PageRank and route "
           "counts from the GraphFrames network are model inputs too.")
ec = st.columns([1, 1, 1, 1, 1])
for col, (label, fn) in zip(ec[:2], EXAMPLES):
    col.button(label, on_click=example, args=(fn,), width="stretch", key=f"ex_{label}")
k = st.columns(5)
k[0].metric("Flights analysed (20% sample)", f"{D['totals']['flights']:,}")
k[1].metric("Arrived 15+ min late", f"{AVG}%")
k[2].metric("Airports in one network", D["totals"]["airports"])
k[3].metric("Routes", f"{D['totals']['routes']:,}")
k[4].metric("Random Forest AUC", RES["Random Forest"]["AUC"])

tabs = st.tabs(["Predict", "Airport network", "Delay explorer", "Models", "About"])

# ---------------------------------------------------------------- predict
with tabs[0]:
    left, right = st.columns([1.1, 1])
    with left:
        with st.expander("Flight details", expanded=True):
            c1, c2, c3 = st.columns([5, 1.2, 5])
            fmt = lambda c: f"{c} – {AP[c]['city']}, {AP[c]['state']}"
            codes = sorted(AP)
            c1.selectbox("From", codes, format_func=fmt, key="o")
            c2.write("")
            c2.write("")
            c2.button("⇄", on_click=swap, help="Swap airports", width="stretch")
            c3.selectbox("To", codes, format_func=fmt, key="d")
            st.selectbox("Airline", list(AN), format_func=lambda c: AN[c], key="al")
            st.pills("Month", range(1, 13), format_func=lambda m: MN[m - 1][:3] + ("" if m in IN_DATA else "*"),
                     key="mo", selection_mode="single")
            st.pills("Day of week", range(1, 8), format_func=lambda m: DN[m - 1][:3], key="dw",
                     selection_mode="single")
            st.select_slider("Scheduled departure time", options=list(range(24)),
                             format_func=hl, key="hr")
            st.caption(f"{hl(ss.hr)} ({period(ss.hr)}). * No data for this month.")
            notes = check(ss.o, ss.d, ss.mo, ss.hr)
            if notes:
                st.warning("**Check your inputs**\n\n" + "\n".join(f"- {n}" for n in notes))
            if ss.msg:
                st.info(ss.msg)
            go = st.button("Predict delay risk", type="primary", width="stretch")
    cur = dict(o=ss.o, d=ss.d, al=ss.al, mo=ss.mo, dw=ss.dw, hr=ss.hr)
    if go:
        ss.msg = ""
        dist, known = distance(cur["o"], cur["d"])
        if cur["o"] == cur["d"] or cur["mo"] is None or cur["dw"] is None or dist is None:
            ss.res = None
            st.error("Fix the inputs above before predicting.")
        else:
            x = features(cur, dist)
            ss.res = dict(inp=cur, dist=dist, known=known, rf=rf_score(x), lr=lr_score(x), notes=notes)
    res = ss.res
    with right:
        with st.expander("Result", expanded=True):
            if not res:
                st.write("Fill in the flight and press **Predict delay risk**. Nothing is scored until you do.")
            else:
                i, r = res["inp"], res["rf"]
                if i != cur:
                    st.info("Inputs changed. Press Predict delay risk to update the result.")
                word, col = (("Likely to arrive 15+ minutes late", AMBER) if r >= .5 else
                             ("Likely to arrive on time", TEAL) if r < .4 else ("Close call", "#888"))
                st.markdown(f'<span class="dot" style="background:{AC[i["al"]]}"></span>'
                            f'**{i["o"]} to {i["d"]}, {AN[i["al"]]}**', unsafe_allow_html=True)
                st.markdown(f'<div class="big" style="color:{col}">{word}</div>', unsafe_allow_html=True)
                st.progress(min(1.0, r), text=f"Random Forest score {r:.0%} (50% is the “late” line)")
                m1, m2 = st.columns(2)
                m1.metric("Random Forest", f"{r:.0%}")
                m2.metric("Logistic Regression", f"{res['lr']:.0%}")
                a, b = AP[i["o"]], AP[i["d"]]
                st.caption(f"Distance {res['dist']:,.0f} miles{'' if res['known'] else ' (estimated from coordinates)'}. "
                           f"PageRank: {a['code']} {a['pr']}, {b['code']} {b['pr']}. "
                           f"Routes out of {a['code']}: {a['out']}.")
                if res["notes"]:
                    st.warning(" ".join(res["notes"]))
    if res:
        i = res["inp"]
        x1, x2 = st.columns(2)
        with x1:
            with st.expander("Same trip, every departure hour", expanded=True):
                df = pd.DataFrame({"hour": range(24),
                                   "score": [rf_score(features(i, res["dist"], hr=h)) for h in range(24)]})
                df["label"] = df.hour.map(hl)
                line = alt.Chart(df).mark_line(color=AC[i["al"]], strokeWidth=3).encode(
                    x=alt.X("hour:O", title="Departure hour"), y=alt.Y("score:Q", axis=alt.Axis(format="%"), title="Score"))
                pts = alt.Chart(df).mark_circle(size=90).encode(
                    x="hour:O", y="score:Q", tooltip=["label", alt.Tooltip("score:Q", format=".0%")],
                    color=alt.condition(alt.datum.hour == i["hr"], alt.value("#111"), alt.value(AC[i["al"]])))
                st.altair_chart(line + pts, width="stretch")
        with x2:
            with st.expander("Same trip, every airline", expanded=True):
                df = pd.DataFrame({"code": AL, "name": [AN[c] for c in AL],
                                   "score": [rf_score(features(i, res["dist"], al=c)) for c in AL]})
                bars = alt.Chart(df).mark_bar().encode(
                    y=alt.Y("name:N", sort="x", title=None),
                    x=alt.X("score:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title="Score"),
                    color=alt.Color("code:N", legend=None,
                                    scale=alt.Scale(domain=list(AC), range=list(AC.values()))),
                    opacity=alt.condition(alt.datum.code == i["al"], alt.value(1), alt.value(.55)),
                    tooltip=["name", alt.Tooltip("score:Q", format=".0%")])
                st.altair_chart(bars, width="stretch")

# ---------------------------------------------------------------- network
with tabs[1]:
    st.write("All airports from the GraphFrames network. Larger dots have a higher PageRank.")
    mc, dc = st.columns([2, 1])
    with mc:
        with st.expander("Airport map", expanded=True):
            s1, s2 = st.columns(2)
            sel = s1.selectbox("Find an airport", sorted(AP), format_func=lambda c: f"{c} – {AP[c]['city']}")
            minr = s2.slider("Minimum routes", 0, 150, 0, 5)
            rank = {a["code"]: n + 1 for n, a in enumerate(sorted(D["airports"], key=lambda a: -a["pr"]))}
            ends = {ss.res["inp"]["o"], ss.res["inp"]["d"]} if ss.res else set()
            rows = [dict(a, hl=a["code"] in ends or a["code"] == sel) for a in D["airports"]
                    if a["lat"] is not None and -126 < a["lon"] < -66 and 23 < a["lat"] < 50
                    and (a["out"] >= minr or rank[a["code"]] <= 10 or a["code"] in ends or a["code"] == sel)]
            dm = pd.DataFrame(rows)
            enc = dict(x=alt.X("lon:Q", scale=alt.Scale(domain=[-125, -66]), axis=None),
                       y=alt.Y("lat:Q", scale=alt.Scale(domain=[24, 50]), axis=None))
            dots = alt.Chart(dm).mark_circle(opacity=.75).encode(
                **enc, size=alt.Size("pr:Q", legend=None, scale=alt.Scale(range=[20, 800])),
                color=alt.condition(alt.datum.hl, alt.value(AMBER), alt.value(TEAL)),
                tooltip=["name", "city", "out", "pr"])
            labels = alt.Chart(dm[(dm.code.map(rank) <= 10) | dm.hl]).mark_text(dy=-10, fontSize=11, fontWeight="bold").encode(
                **enc, text="code:N")
            layers = dots + labels
            if ss.res:
                a, b = AP[ss.res["inp"]["o"]], AP[ss.res["inp"]["d"]]
                if a["lat"] is not None and b["lat"] is not None:
                    ln = pd.DataFrame({"lon": [a["lon"], b["lon"]], "lat": [a["lat"], b["lat"]], "g": [1, 1]})
                    layers = alt.Chart(ln).mark_line(color=AMBER, strokeDash=[6, 4], strokeWidth=2.5).encode(**enc, detail="g") + layers
            st.altair_chart(layers.properties(height=440), width="stretch")
    with dc:
        with st.expander("Airport details", expanded=True):
            a = AP[sel]
            st.subheader(f"{a['name']} ({a['code']})")
            st.write(f"{a['city']}, {a['state']}")
            st.write(f"PageRank rank **#{rank[sel]}** of {len(AP)} (score {a['pr']}).")
            st.write(f"{a['inn']} incoming and {a['out']} outgoing routes.")
            st.caption("The network is one connected component, so every airport can reach every other.")
            st.button("Use as origin", on_click=lambda: ss.update(o=sel), key="uo")
            st.button("Use as destination", on_click=lambda: ss.update(d=sel), key="ud")

# ---------------------------------------------------------------- explorer
with tabs[2]:
    with st.expander("Delays by airline, month and hour", expanded=True):
        view = st.radio("View", ["Airlines", "Months", "Hours"], horizontal=True)
        if view == "Airlines":
            met = st.selectbox("Sort by", ["delayed_pct", "avg_arrival_delay", "flights"],
                               format_func=lambda m: {"delayed_pct": "Delay rate (%)", "avg_arrival_delay":
                                                      "Average delay (min)", "flights": "Flights"}[m])
            df = pd.DataFrame(D["airline_stats"]).rename(columns={"AIRLINE": "code", "AIRLINE_NAME": "name"})
            df["value"] = df[met]
            ch = alt.Chart(df).mark_bar().encode(
                y=alt.Y("name:N", sort="-x", title=None), x=alt.X("value:Q", title=None),
                color=alt.Color("code:N", legend=None, scale=alt.Scale(domain=list(AC), range=list(AC.values()))),
                tooltip=["name", "flights", "delayed_pct", "avg_arrival_delay"])
            st.altair_chart(ch, width="stretch")
            pick = st.selectbox("Details for", df.sort_values("value", ascending=False).code, format_func=lambda c: AN[c])
            r = df[df.code == pick].iloc[0]
            st.success(f"{r['name']}: {int(r['flights']):,} flights, {r['delayed_pct']}% arrived late "
                       f"({abs(r['delayed_pct'] - AVG):.1f} points {'above' if r['delayed_pct'] >= AVG else 'below'} "
                       f"the overall rate), average arrival delay {r['avg_arrival_delay']} minutes.")
        else:
            if view == "Months":
                df = pd.DataFrame(D["by_month"]).rename(columns={"MONTH": "k"})
                df["name"] = df.k.map(lambda m: MN[m - 1])
            else:
                df = pd.DataFrame(D["by_hour"]).rename(columns={"DEP_HOUR": "k"})
                df["name"] = df.k.map(hl)
            df["below"] = df.delayed_pct < AVG
            ch = alt.Chart(df).mark_bar().encode(
                x=alt.X("name:N", sort=list(df.name), title=None), y=alt.Y("delayed_pct:Q", title="Delayed (%)"),
                color=alt.condition(alt.datum.below, alt.value(TEAL), alt.value(AMBER)),
                tooltip=["name", "delayed_pct"])
            st.altair_chart(ch, width="stretch")
            pick = st.selectbox("Details for", list(df.name))
            v = float(df[df.name == pick].delayed_pct.iloc[0])
            st.success(f"{pick}: {v}% of flights arrived late, {abs(v - AVG):.1f} points "
                       f"{'above' if v >= AVG else 'below'} the overall rate.")
        st.caption(f"Delay rate is the share of flights arriving more than 15 minutes late. Overall: {AVG}%.")

# ---------------------------------------------------------------- models
with tabs[3]:
    with st.expander("Compare the models", expanded=True):
        st.dataframe(pd.DataFrame(D["results"]).set_index("Model"), width="stretch")
        C = D["cm"]
        N = sum(C.values())
        base = round((C["tn"] + C["fp"]) / N * 100)
        MD = {"AUC": "How well a model ranks late flights above on-time ones. 0.5 is a coin flip and 1.0 is perfect.",
              "Accuracy": "Share of all flights classed correctly. Class weights trade some accuracy for catching more late flights.",
              "F1": "A balance of precision and recall, averaged over both classes.",
              "Precision": "Of the flights the model put in a class, the share that really belonged there (weighted).",
              "Recall": "Of the flights that really belonged to a class, the share the model found (weighted)."}
        met = st.pills("Metric", list(MD), default="AUC", selection_mode="single") or "AUC"
        dfm = pd.DataFrame({"Model": list(RES), "value": [RES[m][met] for m in RES]})
        best = dfm.loc[dfm.value.idxmax(), "Model"]
        st.altair_chart(alt.Chart(dfm).mark_bar().encode(
            y=alt.Y("Model:N", title=None), x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title=met),
            color=alt.condition(alt.datum.Model == best, alt.value(TEAL), alt.value("#94a3b8"))).properties(height=110),
            width="stretch")
        v = RES[best][met]
        plain = {"AUC": f"Pick one late and one on-time flight at random. {best} gives the late one the higher score about {v:.0%} of the time. Guessing would be 50%.",
                 "Accuracy": f"{best} labelled {v:.0%} of test flights correctly. Always answering “on time” would be right {base}% of the time, but would never catch a late flight.",
                 "F1": f"One number balancing being right when it flags a flight with finding the flights it should flag. {best} scores {v}.",
                 "Precision": f"When {best} puts a flight in a group, it is right about {v:.0%} of the time, averaged over both groups.",
                 "Recall": f"Of the flights that really belonged to a group, {best} found about {v:.0%}, averaged over both groups."}[met]
        st.caption(MD[met] + f" Highest here: {best}.")
        st.info("**In plain words:** " + plain)
        mod = st.radio("How each model works", list(RES), horizontal=True, index=1)
        desc = {"Logistic Regression": "Adds up weighted clues (airline, time, distance, airport traffic) into one score. Fast and easy to explain, but it can only combine clues in a simple additive way.",
                "Random Forest": f"Asks {len(D['rf'])} decision trees to vote, then averages the votes. Each tree is a chain of yes/no questions such as “Is it after 5 pm?” It catches combinations a simple sum misses, such as a busy hub in the evening, but takes longer to train."}[mod]
        gap = abs(RES["Logistic Regression"]["AUC"] - RES["Random Forest"]["AUC"])
        st.write(desc)
        st.caption(f"Training time here: {RES[mod]['Train time (s)']} s. The two models’ AUC differs by only {gap:.3f}.")
    m1, m2 = st.columns(2)
    with m1:
        with st.expander("Random Forest confusion matrix", expanded=True):
            cm = pd.DataFrame([("Actually on time", "Predicted on time", C["tn"]), ("Actually on time", "Predicted late", C["fp"]),
                               ("Actually late", "Predicted on time", C["fn"]), ("Actually late", "Predicted late", C["tp"])],
                              columns=["actual", "pred", "n"])
            h = alt.Chart(cm).encode(x=alt.X("pred:N", title=None, sort=None), y=alt.Y("actual:N", title=None, sort=None))
            st.altair_chart((h.mark_rect().encode(color=alt.Color("n:Q", legend=None, scale=alt.Scale(scheme="blues")))
                             + h.mark_text(fontSize=14).encode(text=alt.Text("n:Q", format=","))).properties(height=220),
                            width="stretch")
            st.caption(f"It caught {C['tp'] / (C['tp'] + C['fn']):.0%} of late flights, but only "
                       f"{C['tp'] / (C['tp'] + C['fp']):.0%} of its late warnings were right.")
    with m2:
        with st.expander("Most important features", expanded=True):
            nice = {"DEP_HOUR": "Departure hour", "MONTH": "Month", "DAY_OF_WEEK": "Day of week", "DISTANCE": "Distance",
                    "ORIGIN_PR": "Origin PageRank", "ORIGIN_IN": "Origin incoming routes", "ORIGIN_OUT": "Origin outgoing routes",
                    "DEST_PR": "Destination PageRank", "DEST_IN": "Destination incoming routes", "DEST_OUT": "Destination outgoing routes"}
            fi = pd.DataFrame([(nice.get(k, "Airline: " + AN.get(k[12:], k[12:])), v, k.startswith(("ORIGIN_", "DEST_")))
                               for k, v in D["importance"].items()], columns=["feature", "importance", "graph"])
            st.altair_chart(alt.Chart(fi).mark_bar().encode(
                y=alt.Y("feature:N", sort="-x", title=None), x="importance:Q",
                color=alt.condition(alt.datum.graph, alt.value(TEAL), alt.value("#94a3b8"))), width="stretch")
            st.caption("Teal bars are graph-derived features.")

# ---------------------------------------------------------------- about
with tabs[4]:
    with st.expander("How a prediction is made", expanded=True):
        top = max(D["airports"], key=lambda a: a["pr"])
        ST = {"1. Load and clean": f"Spark DataFrames read the 2015 flight records and drop cancelled, diverted and incomplete flights. {D['totals']['flights']:,} flights remained, and {AVG}% arrived more than 15 minutes late.",
              "2. Build the network": f"GraphFrames turns airports into nodes and routes into links: {D['totals']['airports']} airports and {D['totals']['routes']:,} routes, all in one connected component.",
              "3. Score the airports": f"PageRank and route counts show how central each airport is. The top hub is {top['name']} ({top['code']}) with {top['out']} outgoing routes.",
              "4. Train the models": f"Spark MLlib trained Logistic Regression and a {len(D['rf'])}-tree Random Forest on 80% of flights, using airline, time, distance and the graph features. Random Forest AUC: {RES['Random Forest']['AUC']}.",
              "5. Run in this app": "The trained trees and weights were exported as JSON and are evaluated here in Python, so predictions need no Spark."}
        step = st.pills("Pipeline stage", list(ST), default="1. Load and clean", selection_mode="single") or "1. Load and clean"
        st.info(ST[step])
    with st.expander("How should I read the score?"):
        st.write("The models were trained with class weights, so the score is not a literal chance of delay. "
                 "A score of 50% or more is where the model predicts “late”. Use it to compare flights and times.")
        sc = st.slider("Try a score", 0, 100, 50, format="%d%%")
        fl, al_, pr = round((C["fp"] + C["tp"]) / N * 100), round((C["fn"] + C["tp"]) / N * 100), round(C["tp"] / (C["tp"] + C["fp"]) * 100)
        st.write(f"The model predicts “late”. At this line it flagged about {fl}% of test flights, while {al_}% were actually late, and about {pr}% of its late warnings were right."
                 if sc >= 50 else "A close call: below the 50% line, so the model does not predict “late”, but it is not clearly on time either."
                 if sc >= 40 else "The model predicts “on time”. Lower scores mean the airline, time and route look better than usual.")
    with st.expander("Where does the data come from?"):
        st.write("U.S. Department of Transportation on-time records for 2015, via "
                 "[Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays). This project used a random 20% sample.")
    with st.expander("What are the limits?"):
        st.write("Only information known before departure is used, with no weather or earlier-flight delays, so accuracy "
                 "is modest (AUC 0.64). It covers 2015 routes only. For a route not in the data, distance is estimated from coordinates.")
