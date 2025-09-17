import streamlit as st
import graphviz
import pandas as pd
from io import BytesIO

# ---------- Helper Functions ----------

def draw_dfa(states, alphabet, transitions, start_state, final_states, title="DFA"):
    dot = graphviz.Digraph(comment=title)
    dot.attr(rankdir="LR", size="8")
    dot.node("", shape="none")

    for state in states:
        if state == start_state and state in final_states:
            dot.node(state, state, shape="doublecircle", style="filled", color="black", fillcolor="white", fontcolor="black")
            dot.edge("", state)
        elif state == start_state:
            dot.node(state, state, shape="circle", style="filled", color="black", fillcolor="white", fontcolor="black")
            dot.edge("", state)
        elif state in final_states:
            dot.node(state, state, shape="doublecircle", style="filled", color="black", fillcolor="white", fontcolor="black")
        else:
            dot.node(state, state, shape="circle", style="filled", color="black", fillcolor="white", fontcolor="black")

    for (src, sym), dst in transitions.items():
        dot.edge(src, dst, label=sym)

    return dot

def export_graph(dot):
    img_bytes = dot.pipe(format="svg")
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
                    key = (p_next, q_next) if p_next < q_next else (q_next, p_next)
                    if key in table and table[key]:
                        new_table[(p, q)] = True
                        changed = True
                        break
        if changed:
            rounds.append(new_table.copy())
        table = new_table
    return rounds

def get_equivalence_classes(table, states):
    """
    Build equivalence classes from the final distinguishability table.
    Preserves the order of states.
    """
    index_map = {s: i for i, s in enumerate(states)}
    groups = []
    visited = set()

    for s in states:   # keep original order
        if s in visited:
            continue
        group = {s}
        visited.add(s)
        for t in states:
            if t not in visited:
                key = (s, t) if index_map[s] < index_map[t] else (t, s)
                if not table.get(key, False):  # not distinguishable → merge
                    group.add(t)
                    visited.add(t)
        groups.append(group)
    return groups

def render_step_down_table(states, current_table, previous_table, round_number):
    st.markdown(f"### 🔢 Step-Down Table – Round {round_number}")

    n = len(states)
    html = "<style> table, th, td {border: 1px solid black; border-collapse: collapse; padding: 8px; text-align: center;} </style>"
    html += "<table>"

    html += "<tr><th></th>"
    for s in states:
        html += f"<th>{s}</th>"
    html += "</tr>"

    for i in range(n):
        html += f"<tr><th>{states[i]}</th>"
        for j in range(n):
            if i <= j:
                html += '<td style="background-color:#ddd;">–</td>'
            else:
                p = states[j]
                q = states[i]
                marked = current_table.get((p, q), False)
                prev_marked = previous_table.get((p, q), False) if previous_table else False

                if marked:
                    cell = '<td style="background-color:#f88;">❌</td>'
                else:
                    cell = '<td style="background-color:#8f8;">✔️</td>'
                html += cell
        html += "</tr>"
    html += "</table>"

    st.markdown(html, unsafe_allow_html=True)

def generate_latex_table(states, alphabet, transitions, start_state, final_states, caption_text):
    rows = []
    for s in states:
        state_label = s
        if s == start_state:
            state_label = "→" + state_label
        if s in final_states:
            state_label = state_label + "*"

        cols = [state_label]
        for a in alphabet:
            dst = transitions.get((s, a), "-")
            cols.append(dst)
        rows.append(" & ".join(cols) + r" \\ \hline")

    latex_code = (
        "\\begin{table}[h]\n"
        "    \\centering\n"
        "    \\begin{tabular}{|" + "c|" * (len(alphabet) + 1) + "}\n"
        "    \\hline\n"
        "    State & " + " & ".join(alphabet) + r" \\ \hline" + "\n"
        + "\n".join(rows) + "\n"
        "    \\end{tabular}\n"
        f"    \\caption{{{caption_text}}}\n"
        "\\end{table}"
    )
    return latex_code

# ---------- Streamlit App ----------

st.set_page_config(page_title="DFA Minimizer", layout="wide")
st.title("🎯 DFA Minimization Visualizer")

# ---------- Input ----------

st.sidebar.header("📥 Input DFA")
uploaded_file = st.sidebar.file_uploader("Upload DFA file", type=["xlsx", "csv"])
dfa_data = {}

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)

        # Replace 'φ', 'ϕ', '-', empty strings with pd.NA
        df = df.replace(["φ", "ϕ", "-", ""], pd.NA)

        st.subheader("📄 Uploaded DFA Table")
        st.dataframe(df)

        # Extract states
        states = []
        start_state = None
        final_states = []
        
        for state_entry in df.iloc[:, 1]:  # second column is "State"
            state_name = state_entry.replace("→", "").replace("*", "").strip()
            states.append(state_name)
            if "→" in state_entry:
                start_state = state_name
            if "*" in state_entry:
                final_states.append(state_name)

        # Extract alphabet (all columns except first two)
        alphabet = list(df.columns[2:])

        # Extract transitions
        transitions = {}
        for _, row in df.iterrows():
            state_entry = row.iloc[1]
            state_name = state_entry.replace("→", "").replace("*", "").strip()
            for a in alphabet:
                val = row[a]
                if pd.notna(val):
                    transitions[(state_name, a)] = str(val).strip()

        dfa_data = {
            'states': states,
            'alphabet': alphabet,
            'start_state': start_state,
            'final_states': final_states,
            'transitions': transitions
        }

    except Exception as e:
        st.error(f"Error reading file: {e}")

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

# ---------- Validation ----------

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

# ---------- Original DFA ----------

st.subheader("📌 Original DFA")
orig_dot = draw_dfa(states, alphabet, transitions, start_state, final_states)
st.graphviz_chart(orig_dot)

orig_img = export_graph(orig_dot)
st.download_button(
    label="📥 Download Original DFA (SVG)",
    data=orig_img,
    file_name="original_dfa.svg",
    mime="image/svg+xml"
)

st.subheader("📋 Original DFA Transition Table (LaTeX)")
latex_original = generate_latex_table(states, alphabet, transitions, start_state, final_states, "Original DFA Transition Table")
st.code(latex_original, language="latex")

# ---------- Minimization Process ----------

st.subheader("✅ Minimization Process")
rounds = refine_table_rounds(states, transitions, alphabet, final_states)
for idx, table in enumerate(rounds):
    previous_table = rounds[idx - 1] if idx > 0 else None
    render_step_down_table(states, table, previous_table, idx)

# ---------- Minimized DFA ----------

st.subheader("🎯 Minimized DFA")
final_table = rounds[-1]
groups = get_equivalence_classes(final_table, states)

# Show merged states info
st.write("**Merged States (Equivalence Classes):**")
for g in groups:
    st.write(" , ".join(sorted(g)))

# Build mapping old → new
index_map = {s: i for i, s in enumerate(states)}
state_mapping = {}
min_states = []
for group in groups:
    if len(group) == 1:
        new_state = next(iter(group))  # keep singleton
    else:
        new_state = "_".join(sorted(group, key=lambda x: index_map[x]))
    min_states.append(new_state)
    for s in group:
        state_mapping[s] = new_state

min_start = state_mapping[start_state]
min_final = sorted({state_mapping[f] for f in final_states})

min_transitions = {}
for (s, a), dst in transitions.items():
    min_transitions[(state_mapping[s], a)] = state_mapping[dst]

min_dot = draw_dfa(min_states, alphabet, min_transitions, min_start, min_final, "Minimized DFA")
st.graphviz_chart(min_dot)

min_img = export_graph(min_dot)
st.download_button(
    label="📥 Download Minimized DFA (SVG)",
    data=min_img,
    file_name="minimized_dfa.svg",
    mime="image/svg+xml"
)

st.subheader("📋 Minimized DFA Transition Table (LaTeX)")
latex_minimized = generate_latex_table(min_states, alphabet, min_transitions, min_start, min_final, "Minimized DFA Transition Table")
st.code(latex_minimized, language="latex")
