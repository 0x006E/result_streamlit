import base64
import logging
import os
from collections import defaultdict
from tempfile import NamedTemporaryFile
from utils.generate_df_filter import generate_dataframe_filter, DataFrameFilterException
from pypdf import PdfReader
import pandas as pd
import plotly.express as px
import streamlit as st
from pdf_parser import parse_pdf
from docxtpl import DocxTemplate
import numpy as np
import io
import asyncio
import pathlib

HERE = pathlib.Path(__file__).parent


st.set_page_config(layout="centered", page_title="Insightify")


def invert_dict(d):
    return {value: key for key, value in d.items()}


def check_dictionary_values(dictionary):
    for key, value in dictionary.items():
        if value is None or value == '':
            return False
    return True

async def parse_pdf_async( pages, final_data={}):
    temp = NamedTemporaryFile(suffix=".pdf", delete=False)
    temp.write(st.session_state.uploaded_file.getvalue())
    result = parse_pdf(file_path=temp.name, pages=pages, final_data=final_data)
    st.success("Processing complete!")
    st.session_state.result = result
    temp.close()
    try:
        os.remove(temp.name)
    except:
        st.warning("Could not delete temporary file")
    next_btn = st.button(
        "Next", key="next", use_container_width=True, on_click=go_next)
    return result


class log_viewer(logging.Handler):
    """ Class to redistribute python logging data """
    # have a class member to store the existing logger
    logger_instance = logging.getLogger("camelot")
    logger = logging.getLogger('pdf_parser')

    def __init__(self, *args, **kwargs):
        # Initialize the Handler
        logging.Handler.__init__(self, *args)
        self.st_container=st.container()
        self.st_container.title("Processing uploaded PDF file")
        self.st_container.write("")
        self.st_instance=self.st_container.empty()
        self.logger.handlers.clear()
        self.logger_instance.handlers.clear()
        formatter = logging.Formatter('%(name)s - %(message)s')
        self.setFormatter(formatter)

        self.logger.setLevel("INFO")
        self.logger_instance.setLevel("INFO")
        self.logger_instance.addHandler(self)
        self.logger.addHandler(self)

    def emit(self, record):
        """ Overload of logging.Handler method """
        record = self.format(record)
        self.st_instance.markdown(f'<div style="text-align: center;font-size:1.5em">{record}</div>', unsafe_allow_html=True)

def go_next():
    st.session_state.page_index += 1


def page_upload():
    def set_file_uploaded():
        st.session_state.uploaded_file = uploaded_file
        go_next()

    st.title("Welcome to Insightify")
    uploaded_file = st.file_uploader(
        "Upload a PDF file", type="pdf", key="upload")

    # Check if a file is uploaded
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
    col1, col2, col3 = st.columns(3)
    next_btn = col2.button(
        "Next", key="next", disabled=uploaded_file is None, on_click=set_file_uploaded, use_container_width=True)


def page_display():
    def go_back():
        st.session_state.page_index = 0
        st.session_state.uploaded_file = None
    e=st.empty()
    c=e.container()
    c.title("This is what you uploaded")

    if "uploaded_file" not in st.session_state or st.session_state.uploaded_file is None:
        c.error("No file uploaded yet.")
        c.stop()

    # Extract text from the uploaded \PDF file
    uploaded_file = st.session_state.uploaded_file
    reader = PdfReader(st.session_state.uploaded_file)
    num_pages = len(reader.pages)
    c.write(f"The PDF contains {num_pages} pages.")
    col1, col2 = c.columns(2)
    st.session_state.start_page = col1.number_input(
        "Start page", min_value=1, max_value=num_pages, step=1)
    st.session_state.end_page = col2.number_input(
        "End page", min_value=1, value=num_pages,  max_value=num_pages, step=1)

    # Convert the PDF to bytes
    pdf_bytes = uploaded_file.getvalue()

    # Embed the PDF file using the ID
    c.write("Here is the uploaded PDF file:")
    c.write(
        f'<iframe src="data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}"  width="700" height="1000" type="application/pdf">', unsafe_allow_html=True)

    # Additional information about the PDF
    c.write(f"The PDF contains {num_pages} pages.")
    col1, col2 = c.columns(2)
    prev_btn = col1.button("Go back", key="prev",
                           disabled=uploaded_file is None, use_container_width=True, on_click=go_back)
    def next_page():
        e.empty()
        go_next()
    next_btn = col2.button(
        "Next", key="next2", disabled=prev_btn, use_container_width=True, on_click=next_page)

    
def empty_page():
    st.write("")
    go_next()

def process_page():
    start_page = st.session_state.start_page
    end_page = st.session_state.end_page
    log_viewer()
    asyncio.run(parse_pdf_async(pages=f"{start_page}-{end_page}", final_data={}))
    

def page_result():
    def go_back():
        st.session_state.page_index = 0
        st.session_state.uploaded_file = None

    st.title("We have some questions for you...")
    if "result" not in st.session_state:
        st.error("No result available yet.")
        st.stop()
    result = st.session_state.result
    if result == {}:
        st.error("This PDF does not seem to contain any result data")
        st.button("Go back", on_click=go_back)
        st.stop()
    st.write("We found these batches in the PDF file:")
    st.markdown('\n'.join(f"- {'20' + i[1:-1]}" for i in result['years']))
    branches = list(set([a for i in result['years']
                    for a in result['years'][i]['branches'].keys()]))
    years = [int(i[1:-1]) for i in result['years'].keys()]
    latest_year = max(years)
    st.info(f"Choosing year 20{latest_year} to source subjects")
    if "branches" not in st.session_state:
        st.session_state.branches = []
    st.session_state.branches = st.multiselect(
        "Choose the branches you want to include in the result:", branches, default=st.session_state.branches)
    if len(st.session_state.branches) == 0:
        st.error("Please select at least one branch")
        st.stop()
    st.subheader(
        "Add credit points to the subjects to display GPA in the result")
    if "scp" not in st.session_state:
        st.session_state.scp = defaultdict(lambda: 0)
    subjects = []
    for branch in st.session_state.branches:
        try:
            subjects += result['years'][f"'{latest_year}'"]['branches'][branch]['students'][0]['subjects'].keys()
        except:
            pass
    st.info(
        "If the credit point is not changed from default value, it will cause incorrect GPA calculation")
    subjects = list(set(subjects))
    with st.form("credit_points"):
        subject_state = defaultdict(lambda: 0)
        if subjects == []:
            st.warning(
                "Subject list seems to be empty, GPA Calculation not possible ")
        for subject in subjects:
            subject_state[subject] = int(st.number_input(
                f"Credit points for {subject } - {result['subjects'][subject]}", min_value=0, max_value=10, value=st.session_state.scp[subject], step=1, key=subject))

        submitted = st.form_submit_button("View results",)
    if submitted:
        st.session_state.scp = subject_state
        go_next()
        st.experimental_rerun()


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
    st.session_state.latest_year = max([int(i[1:-1]) for i in result['years']])
    latest_year = st.session_state.latest_year

    def calculate_sgpa(student):
        sgpa = 0.0
        credits = 0
        for subject in student['subjects']:
            if student['subjects'][subject] == 'Withheld' or student['subjects'][subject] == 'TBP':
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
            student_data['register_info_serial'] = int(
                student['register_info']['serial'])
            for subject in student['subjects']:
                student_data[subject] = student['subjects'][subject]
            if year == f"'{latest_year}'":
                student_data['SGPA'] = calculate_sgpa(student)
                student_data['Full pass'] = not any([True if student['subjects'][subject] in [
                    'FE', 'F', 'Withheld', 'Absent', 'TBP'] else False for subject in student['subjects']])
            data.append(student_data)
        return data

    def go_back():
        st.session_state.page_index -= 1
    tabs = st.tabs(["20"+i[1:-1] for i in result['years']][::-1])
    for tab, year in zip(tabs, reversed(result['years'])):
        with tab:
            branch_tabs = st.tabs(branches)
            for branch_tab, branch in zip(branch_tabs, branches):
                def function():
                    try:
                        with branch_tab:
                            transformed_data = transform_data(
                                result, branch, year)
                            df = pd.DataFrame(
                                data=transformed_data).set_index("Register Number")
                            input = st.text_input(
                                label="Filter criteria", key=year+branch, help="Eg: 30,62..118 includes 30, the numbers from 62 to 118 both inclusive", placeholder="Eg: 30,62..118")
                            try:
                                filter = generate_dataframe_filter(
                                    input, field_name="register_info_serial")
                                if filter is None:
                                    filtered_df = df
                                else:
                                    filtered_df = df.query(
                                        filter)
                            except DataFrameFilterException as e:
                                st.error(str(e))
                                filtered_df = df
                            if year == f"'{latest_year}'":
                                pie_df = filtered_df.groupby(
                                    'Full pass').nunique()
                                length = len(pie_df.index)
                                if length > 1:
                                    values = ['Failed', 'Passed']
                                else:
                                    if df["Full pass"][0]:
                                        values = ["Passed"]
                                    else:
                                        values = ['Failed']
                                pie_df.insert(0, 'Status', values)
                                fig = px.pie(
                                    pie_df, values="register_info_serial", names="Status", title="Full pass percentage")
                                st.plotly_chart(fig, use_container_width=True)
                            filtered_df = filtered_df.sort_values('register_info_serial').drop(
                                ["register_info_serial"], axis=1).reset_index()
                            col1, col2 = st.columns(2)
                            col1.download_button(
                                "Export CSV",
                                convert_df(filtered_df),
                                "file.csv",
                                "text/csv",
                                use_container_width=True,
                                key='download-csv'+year+branch
                            )

                            def generate_docx():
                                st.session_state.filtered_df = filtered_df
                                go_next()
                            col2.button("Generate report", on_click=generate_docx,
                                        key=f"generate-report-{branch}-{year}", use_container_width=True)
                            st.dataframe(
                                filtered_df,
                                use_container_width=True,
                            )

                    except KeyError:
                        pass
                function()
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


def template_generation(context):
    doc = DocxTemplate(str(HERE.joinpath("template.docx")))
    doc.render(context)
    file_stream = io.BytesIO()
    doc.save(file_stream)
    file_stream.seek(0)
    return file_stream


def report_generation():
    def go_back():
        st.session_state.page_index -= 1
        st.session_state.filtered_df = None
    st.title("Report Generation")
    if "filtered_df" not in st.session_state:
        st.error("No view selected")
        st.button("Go back", on_click=go_back)
        st.stop()
    df: pd.DataFrame = st.session_state.filtered_df
    result = st.session_state.result
    col1, col2 = st.columns(2)
    dept = col1.text_input(label="Department name")
    year = col2.number_input(label="Year of admission",
                             value=2000+st.session_state.latest_year)
    staffadvisor1 = col1.text_input(label="Staff Advisor 1")
    staffadvisor2 = col2.text_input(label="Staff Advisor 2")
    semester = col1.text_input(label="Semester")
    date = col2.date_input(label="Date of publication of result")
    st.divider()
    col1, col2 = st.columns(2)
    witheld = col1.number_input(
        label="No. of students whose results are withheld", min_value=0, step=1, value=(df == 'Withheld').any(axis=1).sum())
    appearedinall = col2.number_input(
        label="No. of students appeared in all subjects", min_value=0, step=1, value=(~df.isin(['Absent'])).all(axis=1).sum())
    num_result_published = col1.number_input(
        label="No. of students whose results published", min_value=0, step=1, value=(~df.isin(['Withheld', 'TBP'])).all(axis=1).sum())
    fullpass = col2.number_input(
        label="No. of students passed in all subjects", min_value=0, step=1, value=df['Full pass'].value_counts()[1])
    fullpass_percentage = st.number_input(label="Pass percentage", value=100*df['Full pass'].value_counts()[
                                          1]/df['Full pass'].value_counts().sum())
    st.divider()
    staff_details = st.file_uploader(
        "Staff details", type="csv", key="upload")
    docx = ""
    valid_form = False
    if staff_details is None:
        st.error("Upload staff details")
    else:
        staff_details_df = pd.read_csv(staff_details)
        col1, col2 = st.columns(2)
        subject_code = col1.selectbox(
            "Subject Code", options=staff_details_df.columns)
        name_col = col1.selectbox(
            "Staff Name", options=staff_details_df.drop(subject_code, axis=1).columns)
        subjects = df.columns.values[np.invert(np.in1d(df.columns.values, [
            "Register Number", "Full pass", "SGPA", "register_info_serial"]))]
        staff_details_df.set_index(subject_code, inplace=True)
        col2.write(staff_details_df)
        subject_arr = []
        passed_grades = ['S', "A+", "A", "B+", "B", "C", "C+", "D", "P"]

        for index, subject in enumerate(subjects):
            grades_with_fe = passed_grades + ["FE", "Absent"]
            grades_with_f = grades_with_fe + ['F']
            pass_without_fe = df[subject].isin(passed_grades).sum()
            pass_with_fe = df[subject].isin(grades_with_fe).sum()
            num_result = df[subject].isin(grades_with_f).sum()
            total_without_withheld = df[subject].isin(grades_with_f).sum()
            try:
                facultyname = staff_details_df.loc[subject][name_col]
            except KeyError:
                st.warning(
                    f"Cannot find faculty name for {subject}. Setting as blank")
                facultyname = ""
            sub = {
                'id': index,
                'code': f"{subject} - {result['subjects'][subject]}",
                'faculty': facultyname,
                'num_reg': len(df.index),
                'num_result': num_result,
                'num_passed': pass_without_fe,
                'passwithfe': '{0:.2f}'.format(100*pass_with_fe/total_without_withheld),
                'passwithoutfe': '{0:.2f}'.format(100*pass_without_fe/total_without_withheld)
            }
            subject_arr.append(sub)
        column_headers_dict = {
            'id': 'Sl No.',
            'code': 'Subject with Code',
            'faculty': 'Name of Faculty',
            'num_reg': 'No. of Students Registered',
            'num_result': 'No. of Students whose results published',
            'num_passed': 'No. of Students Passed',
            'num_passwithfe': 'Pass % With FE and Absentees',
            'num_passwithoutfe': 'Pass % without FE and Absentees',
        }
        editable_df = pd.DataFrame(subject_arr).set_index("id")
        renamed_df = editable_df.rename(index=str, columns=column_headers_dict)
        st.write(
            "Below table is editable and may be edited to change incorrect values")
        editable_df = st.data_editor(renamed_df).rename(
            index=str, columns=invert_dict(column_headers_dict))
        context = {
            'dept': dept,
            'year': year,
            'staff_advisor1': staffadvisor1,
            'staff_advisor2': staffadvisor2,
            'withheld': witheld,
            'appearedinall': appearedinall,
            'semester': semester,
            'date': date,
            'num_result_published': num_result_published,
            'fullpass': fullpass,
            'fullpass_percentage': '{0:.2f}%'.format(fullpass_percentage),
            'subjects': subject_arr,
        }
        valid_form = check_dictionary_values(context)
        docx = template_generation(context)
    if not valid_form:
        st.error("Blank fields detected. All fields are required")
    col1, col2 = st.columns(2)
    col1.download_button("Generate", file_name=dept+str(year)+'.docx', data=docx,
                         disabled=staff_details is None or not valid_form, use_container_width=True)
    col2.button("Go back", on_click=go_back, use_container_width=True)


# Initialize app state
if "page_index" not in st.session_state:
    st.session_state.page_index = 0

# Define page navigation
pages = {
    0: page_upload,
    1: page_display,
    3: empty_page,
    2: process_page,
    3: page_result,
    4: page_display_table,
    5: report_generation
}

# Sidebar navigation
pages[st.session_state.page_index]()
