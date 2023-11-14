
import os
import glob
import csv
import shutil
def main():
    path="/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/src/process_benchmarks/summary/merge_summary"
    folder_name="03_track_generated_train_1_10000"
    merged_folder_name = os.path.join(path, folder_name + "_summary")
    if os.path.exists(merged_folder_name):
        shutil.rmtree(merged_folder_name)




    all_folders=os.listdir(path)
    print(len(all_folders),all_folders)


    csv_file_dict = {os.path.basename(csv_file).replace(os.path.basename(csv_file)[os.path.basename(csv_file).find("divided"):os.path.basename(csv_file).find("summary")],""):[] for csv_file in glob.glob(os.path.join(path,all_folders[0])+"/*")}

    print(len(csv_file_dict),csv_file_dict)

    #read divided csv to memory
    for i,folder in enumerate(all_folders):
        current_folder_name=os.path.join(path,folder)
        for csv_file in glob.glob(current_folder_name+"/*"):
            with open(csv_file, 'r') as file:
                one_csv=list(csv.reader(file))
            base_name=os.path.basename(csv_file)
            _replace=base_name[base_name.find("divided"):base_name.find("summary")]
            key=os.path.basename(csv_file).replace(_replace,"")
            csv_file_dict[key].append(one_csv)


    #collect divided tables in memory
    merged_csv_file_dict={os.path.basename(csv_file).replace(os.path.basename(csv_file)[os.path.basename(csv_file).find("divided"):os.path.basename(csv_file).find("summary")],""):[] for csv_file in glob.glob(os.path.join(path,all_folders[0])+"/*")}

    for k,one_file in csv_file_dict.items():
        print(k)
        print(one_file)
        first_row = []
        second_rows=[]
        further_rows=[]
        for rows in one_file:
            first_row=rows[0]
            second_rows.append([rows[1]])
            further_rows.append(rows[2:])




        reconstructed_second_rows=[]
        second_row_tail_list=[]
        for row in second_rows:
            reconstructed_second_rows.append([row[0][:4]])
            second_row_tail_list.append(row[0][4:])
        second_row_tail = [str(sum(int(x) for x in group)) for group in zip(*second_row_tail_list)]
        reconstructed_second_rows[0][0].extend(second_row_tail)


        merged_csv_file_dict[k].append(first_row)
        for r in reconstructed_second_rows:
            merged_csv_file_dict[k].extend(r)
        for r in further_rows:
            merged_csv_file_dict[k].extend(r)

    #create merge folder
    if not os.path.exists(merged_folder_name):
        os.mkdir(merged_folder_name)


    #write merged table to file
    for k,v in merged_csv_file_dict.items():
        with open(os.path.join(merged_folder_name,k), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(v)





if __name__ == '__main__':
    main()