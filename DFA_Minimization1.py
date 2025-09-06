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
            dot.node(state, state, shape="circle", style="filled", color="black", fillcolor="white", fontcolor="black")
        elif state in final_states:
            dot.node(state, state, shape="doublecircle", style="filled", color="black", fillcolor="white", fontcolor="black")
        else:
            dot.node(state, state, shape="circle", style="filled", color="black", fillcolor="white", fontcolor="black")
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
    states = [str(s) for s in st.sidebar.text_input("States (comma separated)", "q0,q1").split(",")]
    alphabet = [str(a) for a in st.sidebar.text_input("Alphabet (comma separated)", "a,b").split(",")]
    start_state = str(st.sidebar.text_input("Start State", "q0"))
    final_states = [str(f) for f in st.sidebar.text_input("Final States (comma separated)", "q1").split(",")]
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

# ---------- Select Download Format ----------
st.sidebar.header("💾 Download Options")
download_format = st.sidebar.selectbox("Select DFA Image Format", options=["png", "pdf", "svg"])

# ---------- Original DFA ----------
st.subheader("📌 Original DFA")
orig_dot = draw_dfa(states, alphabet, transitions, start_state, final_states)
st.graphviz_chart(orig_dot)

# Download button for original DFA
orig_img = export_graph(orig_dot, format=download_format)
st.download_button(
    label=f"📥 Download Original DFA ({download_format.upper()})",
    data=orig_img,
    file_name=f"original_dfa.{download_format}",
    mime=f"image/{download_format}"
)

# ---------- Minimized DFA ----------
st.subheader("✅ Minimized DFA")
rounds = refine_table_rounds(states, transitions, alphabet, final_states)
final_table = rounds[-1]
groups = get_equivalence_classes(final_table, states)

# Create new state names q0, q1, q2...
state_mapping = {}
ordered_groups = []
# Ensure start state group is first
for g in groups:
    if start_state in g:
        ordered_groups = [g] + [x for x in groups if x != g]
        break

for idx, g in enumerate(ordered_groups):
    state_mapping["".join(sorted(g))] = f"q{idx}"

min_states = list(state_mapping.values())
min_start = "q0"
min_final = [state_mapping["".join(sorted(g))] for g in groups if any(f in g for f in final_states)]

# Build minimized transitions
min_transitions = {}
for g in groups:
    rep = next(iter(g))
    new_state = state_mapping["".join(sorted(g))]
    for a in alphabet:
        dst = transitions.get((rep, a))
        if dst:
            for h in groups:
                if dst in h:
                    min_transitions[(new_state, a)] = state_mapping["".join(sorted(h))]

# Draw minimized DFA
min_dot = draw_dfa(min_states, alphabet, min_transitions, min_start, min_final, "Minimized DFA")
st.graphviz_chart(min_dot)

# Download button for minimized DFA
min_img = export_graph(min_dot, format=download_format)
st.download_button(
    label=f"📥 Download Minimized DFA ({download_format.upper()})",
    data=min_img,
    file_name=f"minimized_dfa.{download_format}",
    mime=f"image/{download_format}"
)

# ---------- LaTeX Export for Minimized DFA ----------
st.subheader("📋 Minimized DFA as LaTeX")

rows = []
for s in min_states:
    state_label = s
    if s == min_start:
        state_label = "→" + state_label
    if s in min_final:
        state_label = state_label + "*"

    cols = [state_label]
    for a in alphabet:
        dst = min_transitions.get((s, a), "-")
        cols.append(dst)
    rows.append(" & ".join(cols) + r" \\ \hline")

latex_minimized = (
    "\\begin{table}[h]\n"
    "    \\centering\n"
    "    \\begin{tabular}{|c|" + ("c|" * len(alphabet)) + "}\n"
    "    \\hline\n"
    "    State & " + " & ".join(alphabet) + r" \\ \hline" + "\n"
    + "\n".join(rows) + "\n"
    "    \\end{tabular}\n"
    "    \\caption{Minimized DFA Transition Table}\n"
    "\\end{table}\n"
)

st.code(latex_minimized, language="latex")

# Copy button (safe escaping)
escaped_text = json.dumps(latex_minimized)
copy_button_html = f"""
    <button onclick="navigator.clipboard.writeText({escaped_text})"
            style="padding:6px 10px; border-radius:6px;">
        📋 Copy Minimized DFA (LaTeX)
    </button>
"""
st.markdown(copy_button_html, unsafe_allow_html=True)
