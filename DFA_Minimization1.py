import streamlit as st
import graphviz
import pandas as pd
from io import BytesIO
import json

# ---------- Helper Functions ----------
def draw_dfa(states, alphabet, transitions, start_state, final_states, title="DFA"):
    dot = graphviz.Digraph(comment=title)
    dot.attr(rankdir="LR", size="8")
    dot.node("", shape="none")
    for state in states:
        if state == start_state:
            dot.node(state, state, shape="circle", style="filled", color="red", fontcolor="white")
        elif state in final_states:
            dot.node(state, state, shape="doublecircle", style="filled", color="green", fontcolor="black")
        else:
            dot.node(state, state, shape="circle", style="filled", color="yellow", fontcolor="black")
    dot.edge("", start_state)
    for (src, sym), dst in transitions.items():
        dot.edge(src, dst, label=sym)
    return dot

def export_graph(dot, format="png"):
    """Convert Graphviz dot to an image and return BytesIO buffer."""
    img_bytes = dot.pipe(format=format)
    return BytesIO(img_bytes)

def construct_initial_empty_table(states):
    table = {}
    for i, p in enumerate(states):
        for q in states[i+1:]:
            table[(p, q)] = False
    return table

def refine_table_rounds(states, transitions, alphabet, final_states):
    rounds = []
    round0 = construct_initial_empty_table(states)
    rounds.append(round0)
    round1 = {}
    for i, p in enumerate(states):
        for q in states[i+1:]:
            round1[(p, q)] = (p in final_states) ^ (q in final_states)
    rounds.append(round1)
    table = round1.copy()
    changed = True
    while changed:
        changed = False
        new_table = table.copy()
        for (p, q), marked in list(table.items()):
            if not marked:
                for a in alphabet:
                    p_next = transitions.get((p, a))
                    q_next = transitions.get((q, a))
                    if not p_next or not q_next:
                        continue
                    if (min(p_next, q_next), max(p_next, q_next)) in table:
                        if table[(min(p_next, q_next), max(p_next, q_next))]:
                            new_table[(p, q)] = True
                            changed = True
                            break
        if changed:
            rounds.append(new_table.copy())
        table = new_table
    return rounds

def get_equivalence_classes(table, states):
    groups = []
    ungrouped = set(states)
    while ungrouped:
        s = ungrouped.pop()
        group = {s}
        for t in list(ungrouped):
            if not table.get((min(s, t), max(s, t)), False):
                group.add(t)
                ungrouped.remove(t)
        groups.append(group)
    return groups

def render_stepdown_matrix(states, table, round_num, prev_table=None):
    n = len(states)
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i <= j:
                row.append("–")
            else:
                p, q = states[i], states[j]
                val = table.get((min(p, q), max(p, q)), False)
                row.append("❌" if val else "✅")
        matrix.append(row)
    df = pd.DataFrame(matrix, index=states, columns=states)
    def highlight_cells(val, i, j):
        if val == "–":
            return "background-color: lightgrey; color: grey;"
        if val == "❌" and prev_table is not None:
            p, q = states[i], states[j]
            was_marked = prev_table.get((min(p, q), max(p, q)), False)
            if not was_marked:
                return "background-color: rgba(255,0,0,0.2);"
        return ""
    styled = df.style.apply(
        lambda row: [highlight_cells(row[j], states.index(row.name), j) for j in range(len(row))],
        axis=1
    )
    st.markdown(f"### 🌀 Step-Down Table – Round {round_num}")
    st.dataframe(styled, use_container_width=True)

def copy_button(text, label="📋 Copy LaTeX"):
    button_html = f"""
        <button onclick="navigator.clipboard.writeText(`{text.replace("`", "'")}`)">
            {label}
        </button>
    """
    st.markdown(button_html, unsafe_allow_html=True)

# ---------- Streamlit App ----------
st.set_page_config(page_title="DFA Minimizer", layout="wide")
st.title("🎯 DFA Minimization Visualizer")

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Excel Upload ----------
st.sidebar.header("📥 Excel Upload / Sample")
uploaded_file = st.sidebar.file_uploader("Upload DFA Excel file", type=["xlsx"])
dfa_data = {}

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.subheader("📄 Uploaded Excel Data")
        st.dataframe(df)
        states = [str(s) for s in df['State'].unique()]
        alphabet = [str(a) for a in df['Input'].unique()]
        start_state = str(df['Start_State'].dropna().iloc[0]) if 'Start_State' in df.columns else states[0]
        final_states = [str(f) for f in df['Final_State'].dropna().tolist()] if 'Final_State' in df.columns else []
        transitions = {(str(row['State']), str(row['Input'])): str(row['Next_State']) for _, row in df.iterrows()}
        dfa_data = {'states': states, 'alphabet': alphabet, 'start_state': start_state,
                    'final_states': final_states, 'transitions': transitions}
    except Exception as e:
        st.error(f"Error reading Excel: {e}")

# ---------- Manual Input Fallback ----------
if not dfa_data:
    st.sidebar.header("DFA Input (Manual)")
    states = [str(s) for s in st.sidebar.text_input("States (comma separated)", "A,B,C,D,E,F").split(",")]
    alphabet = [str(a) for a in st.sidebar.text_input("Alphabet (comma separated)", "0,1").split(",")]
    start_state = str(st.sidebar.text_input("Start State", "A"))
    final_states = [str(f) for f in st.sidebar.text_input("Final States (comma separated)", "B,C,F").split(",")]
    st.sidebar.markdown("#### Transitions")
    transitions = {}
    for s in states:
        for a in alphabet:
            dst = st.sidebar.text_input(f"δ({s}, {a})", "")
            if dst:
                transitions[(s, a)] = str(dst)

# ---------- Validate DFA ----------
error_msg = None
if start_state not in states:
    error_msg = f"❌ Start state `{start_state}` is not in states!"
elif not set(final_states).issubset(set(states)):
    error_msg = "❌ Some final states are not in the set of states!"
elif any(dst not in states for dst in transitions.values()):
    error_msg = "❌ Some transitions point to states not in the state set!"

if error_msg:
    st.error(error_msg)
    st.stop()

# ---------- Layout ----------
st.subheader("📌 Original DFA")
orig_dot = draw_dfa(states, alphabet, transitions, start_state, final_states)
st.graphviz_chart(orig_dot)

# ---------- Minimized DFA ----------
st.subheader("✅ Minimized DFA")
rounds = refine_table_rounds(states, transitions, alphabet, final_states)
final_table = rounds[-1]
groups = get_equivalence_classes(final_table, states)

# Remove commas when merging states
min_states = ["".join(sorted(g)) for g in groups]
min_start = next(s for s in min_states if start_state in s)
min_final = [s for s in min_states if any(f in s for f in final_states)]
min_transitions = {}
for group in groups:
    rep = next(iter(group))
    new_state = "".join(sorted(group))
    for a in alphabet:
        dst = transitions.get((rep, a))
        if dst:
            for g in groups:
                if dst in g:
                    min_transitions[(new_state, a)] = "".join(sorted(g))

min_dot = draw_dfa(min_states, alphabet, min_transitions, min_start, min_final, "Minimized DFA")
st.graphviz_chart(min_dot)

# ---------- LaTeX Export for Minimized DFA ----------
st.subheader("📋 Minimized DFA as LaTeX")

# Sort transitions alphabetically by state, then input
sorted_transitions = sorted(min_transitions.items(), key=lambda x: (x[0][0], x[0][1]))

latex_minimized = r"""
\begin{array}{c|c|c}
State & Input & Next State \\
\hline
""" + "\n".join([f"{s} & {a} & {dst} \\\\" for (s, a), dst in sorted_transitions]) + "\n\\end{array}"

st.latex(latex_minimized)
st.code(latex_minimized, language="latex")
copy_button(latex_minimized, "📋 Copy Minimized DFA (LaTeX)")
