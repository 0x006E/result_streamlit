import base64
import logging
import os
from collections import defaultdict
from tempfile import NamedTemporaryFile

import pandas as pd
import plotly.express as px
import PyPDF2
import streamlit as st
from pdf_parser import parse_pdf


class log_viewer(logging.Handler):
    """ Class to redistribute python logging data """

    # have a class member to store the existing logger
    logger_instance = logging.getLogger("pdf_parser")
    st_instance = st.empty()

    def __init__(self, st_instance=None, *args, **kwargs):
        # Initialize the Handler
        logging.Handler.__init__(self, *args)
        st_instance = st_instance or st.empty()

        self.logger_instance.addHandler(self)

    def emit(self, record):
        """ Overload of logging.Handler method """

        record = self.format(record)
        self.st_instance.write(record)
        print(record)


def go_next():
    st.session_state.page_index += 1


def page_upload():
    def set_file_uploaded():
        st.session_state.uploaded_file = uploaded_file
        go_next()
    st.title("Upload source result PDF file")
    uploaded_file = st.file_uploader(
        "Upload a PDF file", type="pdf", key="upload")

    # Check if a file is uploaded
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
    next_btn = st.button(
        "Next", key="next", disabled=uploaded_file is None, on_click=set_file_uploaded)


def page_display():
    def go_back():
        st.session_state.page_index = 0
        st.session_state.uploaded_file = None

    st.title("This is what you uploaded")

    if "uploaded_file" not in st.session_state or st.session_state.uploaded_file is None:
        st.error("No file uploaded yet.")
        st.stop()

    # Extract text from the uploaded \PDF file
    uploaded_file = st.session_state.uploaded_file
    pdf_reader = PyPDF2.PdfFileReader(st.session_state.uploaded_file)
    num_pages = pdf_reader.numPages
    st.write(f"The PDF contains {num_pages} pages.")

    # Convert the PDF to bytes
    pdf_bytes = uploaded_file.getvalue()

    # Embed the PDF file using the ID
    st.write("Here is the uploaded PDF file:")
    st.write(
        f'<iframe src="data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}"  width="700" height="1000" type="application/pdf">', unsafe_allow_html=True)

    # Additional information about the PDF
    st.write(f"The PDF contains {num_pages} pages.")
    col1, col2 = st.columns(2)
    prev_btn = col1.button("Go back", key="prev",
                           disabled=uploaded_file is None, use_container_width=True, on_click=go_back)
    next_btn = col2.button(
        "Next", key="next", disabled=prev_btn, use_container_width=True, on_click=go_next)


def process_page():
    st.title("Processing the Uploaded PDF File")
    code_block = st.empty()

    with st.spinner("Processing the uploaded PDF file..."):
        temp = NamedTemporaryFile(suffix=".pdf", delete=False)
        temp.write(st.session_state.uploaded_file.getvalue())
        log_viewer(st_instance=code_block)
        logfile = NamedTemporaryFile(suffix=".log", delete=False)
        st.session_state.result = parse_pdf(temp.name, logfile.name)
        temp.close()
        logfile.close()
        os.remove(temp.name)
    st.success("Processing complete!")
    next_btn = st.button(
        "Next", key="next", use_container_width=True, on_click=go_next)


def page_result():
    st.title("We have some questions for you...")
    if "result" not in st.session_state:
        st.error("No result available yet.")
        st.stop()
    result = st.session_state.result
    st.write("We found these batches in the PDF file:")
    st.markdown('\n'.join(f"- {'20' + i[1:-1]}" for i in result['years']))
    branches = list(set([a for i in result['years']
                    for a in result['years'][i]['branches'].keys()]))
    years = [int(i[1:-1]) for i in result['years'].keys()]
    latest_year = max(years)
    st.info(f"Choosing year 20{latest_year} to source subjects")
    st.session_state.branches = st.multiselect(
        "Choose the branches you want to include in the result:", branches, default=st.session_state.branches if "branches" in st.session_state else [])
    if len(st.session_state.branches) == 0:
        st.error("Please select at least one branch")
        st.stop()
    st.subheader(
        "Add credit points to the subjects to display GPA in the result")
    if "scp" not in st.session_state:
        st.session_state.scp = defaultdict(lambda: 0)
    subjects = []
    for branch in st.session_state.branches:
        subjects += result['years'][f"'{latest_year}'"]['branches'][branch]['students'][0]['subjects'].keys()
    st.info(
        "If the credit point is not changed from default value, it will cause incorrect GPA calculation")
    subjects = list(set(subjects))
    for subject in subjects:
        st.session_state.scp[subject] = st.number_input(
            f"Credit points for {subject } - {result['subjects'][subject]}", min_value=0, max_value=10, value=st.session_state.scp[subject], step=1, key=subject)

    st.button("View results", key="view",
              use_container_width=True, on_click=go_next)


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


def page_display_table():
    st.header("Result")

    result = st.session_state.result
    branches = st.session_state.branches
    st.session_state.gpi = {
        'S': 10,
        'A+': 9,
        'A': 8.5,
        'B+': 8,
        'B': 7.5,
        'C+': 7,
        'C': 6.5,
        'D': 6,
        'P': 5,
        'F': 0,
    }
    latest_year = max([int(i[1:-1]) for i in result['years']])

    def calculate_sgpa(student):
        sgpa = 0.0
        credits = 0
        for subject in student['subjects']:
            if student['subjects'][subject] == 'Withheld' or student['subjects'][subject] == 'TBP' or student['subjects'][subject] == 'Absent':
                return 0.0
            cpi = st.session_state.scp[subject]
            try:
                gpi = st.session_state.gpi[student['subjects'][subject]]
            except KeyError:
                gpi = 0
            sgpa += cpi * gpi
            credits += cpi
        try:
            sgpa /= credits
        except ZeroDivisionError:
            sgpa = 0.0
        return sgpa

    def transform_data(result, branch, year):
        data = []
        for student in result['years'][year]['branches'][branch]['students']:
            student_data = {}
            student_data['Register Number'] = student['register_number']
            for subject in student['subjects']:
                student_data[subject] = student['subjects'][subject]
            if year == f"'{latest_year}'":
                student_data['SGPA'] = calculate_sgpa(student)
                student_data['Full pass'] = not any([True if student['subjects'][subject] in [
                    'F', 'Withheld', 'Absent', 'TBP'] else False for subject in student['subjects']])
            data.append(student_data)
        return data

    def go_back():
        st.session_state.page_index -= 1
    tabs = st.tabs(["20"+i[1:-1] for i in result['years']][::-1])
    for tab, year in zip(tabs, reversed(result['years'])):
        with tab:
            branch_tabs = st.tabs(branches)
            for branch_tab, branch in zip(branch_tabs, branches):
                try:
                    with branch_tab:
                        df = pd.DataFrame(
                            transform_data(result, branch, year),
                        )
                        if year == f"'{latest_year}'":
                            pie_df = df.groupby('Full pass')[
                                'Register Number'].nunique()
                            fig = px.pie(
                                pie_df, values="Register Number", names="Register Number", title="Full pass percentage")
                            st.plotly_chart(fig, use_container_width=True)
                        st.download_button(
                            "Export CSV",
                            convert_df(df),
                            "file.csv",
                            "text/csv",
                            use_container_width=True,
                            key='download-csv'+year+branch
                        )
                        st.dataframe(
                            df,
                            hide_index=True,
                            use_container_width=True,
                        )
                except KeyError:
                    pass
    st.divider()
    st.caption("The following are the subjects and their credit points")
    st.table(
        pd.DataFrame(
            [(subject, result['subjects'][subject], st.session_state.scp[subject])
             for subject in st.session_state.scp],
            columns=['Code', 'Name', 'Credit Points']
        )
    )
    st.button("Go back", key="back",
              use_container_width=True, on_click=go_back)


# Initialize app state
if "page_index" not in st.session_state:
    st.session_state.page_index = 0

# Define page navigation
pages = {
    0: page_upload,
    1: page_display,
    2: process_page,
    3: page_result,
    4: page_display_table
}

# Sidebar navigation
pages[st.session_state.page_index]()
