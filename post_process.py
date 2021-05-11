from utils import compute_output

directory = 'C:\\Users\\agall\\OneDrive\\Desktop\\Results_PV2000\\'

test_list = ['08', '09']
configuration_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09',
                      '10', '11', '12', '13', '14', '15', '16', '17', '18']
name_list = ['episode', 'baseline']

for id_test in test_list:
    for id_configuration in configuration_list:
        for name in name_list:
            compute_output(directory, id_test, id_configuration, name)

