# Command: python clinic_json.py folder_path
# folder/subfolder/txt files output to folder

import os
import re
import json
from datetime import datetime

def parse_time(time_str):
    """Support day/month/year format"""
    try:
        return datetime.strptime(time_str, "%d/%m/%Y %H:%M:%S")
    except ValueError:
        # return datetime.strptime(time_str, "%d/%m/%Y %H:%M")
        return None

def format_timedelta(delta_minutes):
    """Return xx days xx hours xx minutes format based on minutes"""
    days = delta_minutes // (24 * 60)
    hours = (delta_minutes % (24 * 60)) // 60
    minutes = delta_minutes % 60

    parts = []
    if days > 0:
        parts.append(f"{days} days")
    if hours > 0:
        parts.append(f"{hours} hours")
    if minutes > 0 or not parts:
        parts.append(f"{minutes} minutes")

    return ' '.join(parts)

def time_diff_readable(t1_str, t2_str):
    t1 = parse_time(t1_str)
    t2 = parse_time(t2_str)
    if t1 is None or t2 is None:
        return None
    delta = abs(t1 - t2)
    delta_minutes = int(delta.total_seconds() // 60)
    return format_timedelta(delta_minutes)

# Step name mapping
STEP_MAPPING = {
    'Step_admissions': 'patient admission information',
    'Step_chartevents': 'patient vital signs and testing data',
    'Step_diagnoses_icd':'patient diagnosis information',
    'Step_drgcodes':'other information',
    'Step_emar':'patient medication information',
    'Step_icustays':'ICU hospitalization information',
    'Step_ingredientevents':'patient infusion information',
    'Step_inputevents':'patient input treatment information',
    'Step_labevents':'laboratory information',
    'Step_microbiologyevents':'microbiological examination information',
    'Step_outputevents':'output information',
    'Step_prescriptions':'prescription information',
    'Step_procedureevents':'procedure event',
    'Step_procedures_icd':'procedure information',
    'Step_transfers':'patient transfer information',
    'Step_emar_detail':'patient medication details',
}

# Field name mapping
FIELD_TRANSLATION = {
    'admittime': 'admittime',
    'dischtime': 'dischtime',
    'deathtime': 'deathtime',
    'admission_type': 'type of admission',
    'admission_location': 'location of admission',
    'discharge_location': 'location of discharge',
    'insurance': 'payment method',
    'race': 'race',
    'hospital_expire_flag': 'death sign',
    'charttime': 'Clinical recording time',
    'label': 'project name',
    'category': 'project category',
    'unitname': 'unitname',
    'lownormalvalue': 'lownormalvalue',
    'highnormalvalue': 'highnormalvalue',
    'long_title': 'The name of the diagnosis',
    'drg_type': 'drg type',
    'description': 'group description',
    'drg_severity': 'severity grading',
    'drg_mortality': 'classification of mortality risk',
    'emar_seq': 'emar sqp',
    'medication': 'description of the name of the drug',
    'first_careunit': 'first ICU ward',
    'last_careunit': 'last ICU ward',
    'intime': 'ICU transfer time',
    'outtime': 'ICU roll-out time',
    'los': 'length of stay',
    'starttime': 'infusion start time',
    'endtime': 'infusion end time',
    'amount': 'total amount of infusion',
    'amountuom': 'Unit Of Measure',
    'rate': 'Infusion rate',
    'rateuom': 'rate unit',
    'patientweight': 'weight',
    'flag': 'Whether it is abnormal',
    'fluid': 'specimen type',
    'spec_type_desc': 'description of the specimen type',
    'test_name': 'test name',
    'org_name': 'microorganism name',
    'ab_name': 'antibiotic name',
    'interpretation': 'susceptibility results',
    'value': 'test result value',
    'valueuom': 'unit value',
    'stoptime': 'time to stop medication',
    'drug': 'drug name',
    'formulary_drug_cd': 'drug catalog code',
    'dose_val_rx': 'prescription dose values',
    'dose_unit_rx': 'prescription dosage units',
    'route': 'route of administration',
    'location': 'location',
    'locationcategory': 'location category',
    'seq_num': 'seqnum',
    'chartdate': 'date of record',
    'eventtype': 'event type',
    'careunit': 'nursing unit',
    'dose_given': 'actual dose administered',
    'dose_given_unit': 'actual dose units',
    'product_description': 'description of the drug',
    'infusion_rate': 'infusion rate',
    'infusion_rate_unit': 'infusion rate unit',
    'subject_id': 'subject_id',
    'stay_id': 'stay_id',
    'hadm_id': 'hadm_id',
    'itemid': 'itemid'
}

# Format modification
def clean_value(key, value):
    return value.strip()

# Read file
def parse_input(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    sections = re.split(r'====\s*([^=]+)\s*====', content)
    step_data = {}
    for i in range(1, len(sections), 2):
        section_name = sections[i].strip()
        current_section = []
        for line in sections[i + 1].strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            item = {}
            sgl_pair = [pair for segment in line.split(';') for pair in segment.split(',')]
            for pair in sgl_pair:
                if ':' not in pair:
                    continue
                k, v = map(str.strip, pair.split(':', 1))
                translated = FIELD_TRANSLATION.get(k, k)  # Translate
                cleaned_value = clean_value(translated, v)
                item[translated] = cleaned_value
            if item:
                current_section.append(item)
        step_data[section_name] = current_section
    all_data = []
    in_time = step_data['Step_admissions'][0]['admittime']
    for step, data in step_data.items():
        formatted_step = {
            'type': step,
            'name': STEP_MAPPING[step],
            'records': [format_content(item, step, in_time) for item in data]
        }
        all_data.append(formatted_step)
        print(f"There are {len(data)} records in the {step} table") 
    
    return all_data


def format_content(item, source_step, in_time):
    formatted = {}
    if source_step == "Step_admissions":
        in_admittime = item.get('admittime', '')
        in_dischtime = item.get('dischtime', '')
        in_deathtime = item.get('deathtime', '')
        in_admission_type = item.get('type of admission', '')
        in_admission_location = item.get('location of admission', '')
        in_discharge_location = item.get('location of discharge', '')
        in_insurance = item.get('payment method', '')
        in_race = item.get('race', '')
        in_hospital_expire_flag = item.get('death sign', '')
        # Filter out null values
        valid_parts = []
        if in_admittime: valid_parts.append(f"admittime: {in_admittime}")
        if in_dischtime: valid_parts.append(f"dischtime: {in_dischtime}")
        if in_deathtime: valid_parts.append(f"deathtime: {in_deathtime}")
        if in_admission_type: valid_parts.append(f"type of admission: {in_admission_type}")
        if in_admission_location: valid_parts.append(f"location of admission: {in_admission_location}")
        if in_discharge_location: valid_parts.append(f"location of discharge: {in_discharge_location}")
        if in_insurance: valid_parts.append(f"payment method: {in_insurance}")
        if in_race: valid_parts.append(f"race: {in_race}")
        if in_hospital_expire_flag: valid_parts.append(f"death sign: {in_hospital_expire_flag}")
        
        formatted_str = ", ".join(valid_parts)
        formatted = {
            'patient admission information': formatted_str
        }
    elif source_step == "Step_chartevents":  # Direct output
        in_charttime = item.get('Clinical recording time', '')
        in_label = item.get('project name', '')
        in_category = item.get('project category', '')
        in_unitname = item.get('unitname', '')
        in_lownormalvalue = item.get('lownormalvalue', '')
        in_highnormalvalue = item.get('highnormalvalue', '')
        # Filter out null values
        valid_parts = []
        if in_charttime: valid_parts.append(f"Clinical recording time was {in_charttime}")
        if in_label: valid_parts.append(f"project name was {in_label}")
        if in_category: valid_parts.append(f"project category was {in_category}")
        if in_unitname: valid_parts.append(f"the unitname was {in_unitname}")
        if in_lownormalvalue or in_highnormalvalue: 
            valid_parts.append(f"lownormalvalue and highnormalvalue was {in_lownormalvalue} {in_highnormalvalue}")
        
        formatted_str = ", ".join(valid_parts)
        formatted = {
            'patient vital signs and testing data': formatted_str
        }

    elif source_step == "Step_diagnoses_icd":  # Direct output
        in_title = item.get('The name of the diagnosis', '')
        formatted_str = f"The name of the diagnosis was {in_title}"
        formatted = {
            'patient diagnosis information': formatted_str
        }
    elif source_step == "Step_drgcodes":  # Direct output
        in_drg_type = item.get('drg type', '')
        in_description = item.get('group description', '')
        in_drg_severity = item.get('severity grading', '')
        in_drg_mortality = item.get('classification of mortality risk', '')
        formatted_str = f"the drg type was {in_drg_type},roup description: {in_description},and severity grading was {in_drg_severity} classification of mortality risk was {in_drg_mortality}"
        formatted = {
            'other information': formatted_str
        }
    elif source_step == "Step_emar":  # Direct output
        in_emar_seq = item.get('emar seq', '')
        in_charttime1 = item.get('Clinical recording time', '')
        in_medication = item.get('description of the name of the drug', '')
        formatted_str = f"The drug was given for the {in_emar_seq} time and give time was {in_charttime1},give medication was {in_medication}"
        formatted = {
            'patient medication information': formatted_str
        }
    elif source_step == "Step_icustays":  # Direct output
        in_first_careunit = item.get('first ICU ward', '')
        in_last_careunit = item.get('last ICU ward', '')
        in_intime = item.get('ICU transfer time', '')
        in_outtime = item.get('ICU roll-out time', '')
        in_los = item.get('length of stay', '')
        formatted_str = f"the first ICU ward was {in_first_careunit},last ICU ward was {in_last_careunit},ICU transfer time was {in_intime},ICU roll-out time was {in_outtime},the length of stay was {in_los}"
        formatted = {
            'ICU hospitalization information': formatted_str
        }
    elif source_step == "Step_ingredientevents":  # Direct output
        in_starttime = item.get('infusion start time', '')
        in_endtime = item.get('infusion end time', '')
        in_amount = item.get('total amount of infusion', '')
        in_amountuom = item.get('Unit Of Measure', '')
        in_rate = item.get('Infusion rate', '')
        in_rateuom = item.get('rate unit', '')
        formatted_str = f"patient infusion start time was {in_starttime},infusion end time was {in_endtime}, total amount of infusion {in_amount} with unit Of measure was {in_amountuom},Infusion rate was {in_rate} with unit of rate was {in_rateuom}"
        formatted = {
            'patient infusion information': formatted_str
        }
    elif source_step == "Step_inputevents":  # Direct output
        in_starttime1 = item.get('infusion start time', '')
        in_endtime1 = item.get('infusion end time', '')
        in_amount1 = item.get('total amount of infusion', '')
        in_amountuom1 = item.get('Unit Of Measure', '')
        in_rate1 = item.get('Infusion rate', '')
        in_rateuom1 = item.get('rate unit', '')
        in_weight = item.get('weight', '')
        formatted_str = f"patient infusion start time was {in_starttime1},infusion end time was {in_endtime1}, total amount of infusion {in_amount1} with unit Of measure was {in_amountuom1},Infusion rate was {in_rate1} with unit of rate was {in_rateuom1} and the patient weight was {in_weight}"
        formatted = {
            'patient input treatment information': formatted_str
        }
    elif source_step == "Step_labevents":  # Direct output
        in_charttime2 = item.get('Clinical recording time', '')
        in_flag = item.get('Whether it is abnormal', '')
        in_label1 = item.get('project name', '')
        in_fluid = item.get('specimen type', '')
        formatted_str = f"patient clinical recording time was {in_charttime2},Whether it is abnormal: {in_flag},project name was {in_label1} and specimen type was {in_fluid} "
        formatted = {
            'laboratory information': formatted_str
        }
    elif source_step == "Step_microbiologyevents":  # Direct output
        in_charttime3 = item.get('Clinical recording time', '')
        in_spec_type_desc = item.get('description of the specimen type', '')
        in_test_name = item.get('test name', '')
        in_org_name = item.get('microorganism name', '')
        in_ab_name = item.get('antibiotic name', '')
        in_interpretation = item.get('susceptibility results', '')
        formatted_str = f"patient clinical recording time was {in_charttime3},description of the specimen type was {in_spec_type_desc},and test name was {in_test_name},microorganism name was {in_org_name},antibiotic name was {in_ab_name},susceptibility results was {in_interpretation}"
        formatted = {
            'microbiological examination information': formatted_str
        }
    elif source_step == "Step_outputevents":  # Direct output
        in_charttime4 = item.get('Clinical recording time', '')
        in_value = item.get('test result value', '')
        in_valueuom = item.get('unit value', '')
        formatted_str = f"Clinical recording time:{in_charttime4}, test result value:{in_value}, unit value:{in_valueuom},"
        formatted = {
            'output information': formatted_str
        }
    elif source_step == "Step_prescriptions":  # Direct output
        in_starttime2 = item.get('infusion start time', '')
        in_stoptime = item.get('time to stop medication', '')
        in_drug = item.get('drug name', '')
        in_formulary_drug_cd = item.get('drug catalog code', '')
        in_dose_val_rx = item.get('prescription dose values', '')
        in_dose_unit_rx = item.get('prescription dosage units', '')
        in_route = item.get('route of administration', '')
        formatted_str = f"The time of initiation of medication was {in_starttime2} and time to stop medication was {in_stoptime},the drug name was {in_drug},drug catalog code was {in_formulary_drug_cd},prescription dose values was {in_dose_val_rx},prescription dosage units was {in_dose_unit_rx},route of administration was {in_route}"
        formatted = {
            'prescription information': formatted_str
        }
    elif source_step == "Step_procedureevents":  # Direct output
        in_starttime3 = item.get('infusion start time', '')
        in_endtime2 = item.get('infusion end time', '')
        in_value1 = item.get('test result value', '')
        in_valueuom1 = item.get('unit value', '')
        in_location = item.get('location', '')
        in_locationcategory = item.get('location category', '')
        formatted_str = f"treatment start time: {in_starttime3},treatment end time: {in_endtime2},test result value was {in_value1},and unit value was {in_valueuom1},the location was {in_location},location category was {in_locationcategory}"
        formatted = {
            'procedure event': formatted_str
        }
    elif source_step == "Step_procedures_icd":  # Direct output
        in_seq_num = item.get('seqnum', '')
        in_chartdate = item.get('date of record', '')
        in_long_title = item.get('The name of the diagnosis', '')
        formatted_str = f"{in_seq_num} surgery,and the date of record was {in_chartdate},The name of the diagnosis was {in_long_title}"
        formatted = {
            'procedure information': formatted_str
        }
    elif source_step == "Step_transfers":  # Direct output
        in_event_type = item.get('event type', '')
        in_careunit = item.get('nursing unit', '')
        in_intime1 = item.get('ICU transfer time', '')
        in_outtime1 = item.get('ICU roll-out time', '')
        formatted_str = f"the event type was {in_event_type}, and nursing unit was {in_careunit},time to enter the care unit was {in_intime1},Time away from care unit was {in_outtime1}"
        formatted = {
            'patient transfer information': formatted_str
        }
    elif source_step == "Step_emar_detail":  # Direct output
        in_dose_given = item.get('actual dose administered', '')
        in_dose_given_unit = item.get('actual dose units', '')
        in_product_description = item.get('description of the drug', '')
        in_infusion_rate = item.get('infusion rate', '')
        in_infusion_rate_unit = item.get('infusion rate unit', '')
        in_route1 = item.get('route of administration', '')
        formatted_str = f"actual dose administered was {in_dose_given},and actual dose units was {in_dose_given_unit},description of the drug was {in_product_description},infusion rate was {in_infusion_rate} and infusion rate unit was {in_infusion_rate_unit},route of administration was {in_route1}"
        formatted = {
            'patient medication details': formatted_str
        }
    return formatted


def generate_json(data_list_cnt, data, output_path):
    def convert_to_text(data_item, step_name=None):
        if isinstance(data_item, dict):
            # If there is only one key and the key equals step_name, return the value directly
            if len(data_item) == 1 and step_name is not None and step_name in data_item:
                return str(data_item[step_name])
            
            # For chartevents, only output key information
            if step_name == 'Step_chartevents':
                # Only extract key fields
                key_fields = ['Clinical recording time', 'project name', 'project category']
                text_parts = []
                for field in key_fields:
                    if field in data_item and data_item[field]:
                        text_parts.append(f"{field}: {data_item[field]}")
                return " | ".join(text_parts) if text_parts else ""
            
            # Simplify processing for other steps
            text_parts = []
            for key, value in data_item.items():
                if key in ['type', 'name']:
                    continue
                if value and str(value).strip():
                    text_parts.append(str(value))
            return " | ".join(filter(None, text_parts))
        elif isinstance(data_item, list):
            # Simplify list processing
            return " | ".join(filter(None, [str(item) for item in data_item if item]))
        elif data_item and str(data_item).strip():
            return str(data_item)
        return ""

    # Create a list of all admission records - each txt file corresponds to one admission record
    admission_contents = [""] * data_list_cnt

    # First process original data - process txt file data
    for i in range(data_list_cnt):
        step_merge = {}
        for step_data in data[i]:
            if isinstance(step_data, dict):
                step_name = step_data.get('name', '')
                records = step_data.get('records', [])
                
                # Special handling for chartevents data
                if step_name == 'patient vital signs and testing data':
                    # Group and merge chartevents data by time
                    time_groups = {}
                    for record in records:
                        # Extract the inner dictionary
                        actual_record = record.get(step_name, {})
                        if actual_record:
                            # Parse the formatted string to extract time
                            time_match = re.search(r'Clinical recording time was ([^,]+)', actual_record)
                            if time_match:
                                time_key = time_match.group(1).strip()
                                if time_key not in time_groups:
                                    time_groups[time_key] = []
                                time_groups[time_key].append(actual_record)
                    
                    # Merge data from the same time point
                    merged_chartevents = []
                    for time_key, contents in time_groups.items():
                        if len(contents) > 1:
                            # Group by category
                            category_groups = {}
                            for content in contents:
                                # Extract category information
                                category_match = re.search(r'project category was ([^,]+)', content)
                                if category_match:
                                    category = category_match.group(1).strip()
                                    if category not in category_groups:
                                        category_groups[category] = []
                                    category_groups[category].append(content)
                            
                            # Generate summary information
                            summary_parts = [f"Clinical recording time was {time_key}"]
                            for category, items in category_groups.items():
                                if len(items) > 1:
                                    summary_parts.append(f"{category}: {len(items)} measurements")
                                else:
                                    # Single measurement, display specific item
                                    label_match = re.search(r'project name was ([^,]+)', items[0])
                                    if label_match:
                                        summary_parts.append(f"{category}: {label_match.group(1).strip()}")
                            
                            merged_content = ", ".join(summary_parts)
                        else:
                            merged_content = contents[0]
                        merged_chartevents.append(merged_content)
                    
                    if merged_chartevents:
                        step_merge[step_name] = merged_chartevents
                else:
                    # Process other types of data
                    text_contents = [convert_to_text(rec) for rec in records]
                    if any(text_contents):
                        step_merge[step_name] = [content for content in text_contents if content]

        record_parts = []
        for step_name, contents in step_merge.items():
            merged_content = ', '.join(contents)
            record_parts.append(f"{step_name}: {merged_content}")
        if record_parts:
            admission_contents[i] = '\n'.join(record_parts)
            print(f"Length of content for admission {i+1} is {len(admission_contents[i])}")

    json_data = {}
    
    # Merge content of all admission records
    merged_text = ''
    for i, content in enumerate(admission_contents):
        if content.strip():  # Ensure content is not empty
            merged_text += f"The patient's admission was recorded for the {i + 1} time: \n{content}\n\n"
    
    if merged_text.strip():
        json_data['text'] = merged_text.strip()
    else:
        json_data['text'] = "No available records"

    # Write to JSON file
    try:
        # Use the json module to correctly handle JSON format
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            # Ensure the text field is a string, but not additionally quoted in JSON
            json_str = json.dumps(json_data, ensure_ascii=False, indent=None)
            f.write(json_str)
    except Exception as e:
        print(f"Error writing JSON file: {str(e)}")
        print(f"File path: {output_path}")
        print(f"JSON data: {json_data}")
def process_single_file(input_path):
    all_data = parse_input(input_path)
    return all_data

def process_folder(folder_path, folder_path_output):
    for sub_folder in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, sub_folder)
        if os.path.isdir(sub_folder_path):
            all_data_list = []
            data_list_cnt = 0
            file_data_pairs = []  # Store file paths and corresponding data
            
            # Collect all txt files and their data
            for file in os.listdir(sub_folder_path):
                file_path = os.path.join(sub_folder_path, file)
                if file.endswith('.txt'):
                    all_data = process_single_file(file_path)
                    file_data_pairs.append((file_path, all_data))
            
            # Sort admission records by time
            def get_admittime_from_file(file_path):
                """Extract admittime directly from the file"""
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Find the admittime field
                    match = re.search(r'admittime:([^,;]+)', content)
                    if match:
                        return match.group(1).strip()
                except:
                    pass
                return ''
            
            # Sort by admittime
            file_data_pairs.sort(key=lambda x: get_admittime_from_file(x[0]))
            
            # Output the sorted admission times
            print(f"\nAdmission records for {sub_folder} sorted by time:")
            for i, (file_path, all_data) in enumerate(file_data_pairs):
                admittime = get_admittime_from_file(file_path)
                print(f"  Admission {i+1}: {admittime}")
            
            # Process data in sorted order
            for file_path, all_data in file_data_pairs:
                all_data_list.append(all_data)
                data_list_cnt += 1
            
            output_file = os.path.join(folder_path_output, f"{sub_folder}.json")
            generate_json(data_list_cnt, all_data_list, output_file)


if __name__ == "__main__":
    input_folder = 'MIMICtest'
    output_folder = 'samples'
    if not os.path.isdir(input_folder):
        print("Error: Input folder does not exist")
        exit(1)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    process_folder(input_folder, output_folder)