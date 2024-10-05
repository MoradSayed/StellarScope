import csv, os

class CSVReader:
    def __init__(self, file1, file2):
        self.data1 = self.read_csv(file1)
        self.data2 = self.read_csv(file2)

    def read_csv(self, filename):
        data = {}
        with open(filename, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                key = row[reader.fieldnames[0]]
                data[key] = {field: row[field] if row[field] else None for field in reader.fieldnames}
        return data

    def get_row(self, file_number, key):
        if file_number == 1:
            return self.data1.get(key, None)
        elif file_number == 2:
            return self.data2.get(key, None)
        else:
            raise ValueError("Invalid file number. Use 1 or 2.")

path = os.path.join("Assets", "Cache")
csv_reader = CSVReader(os.path.join(path, "stellarhosts_info.csv"), 
                       os.path.join(path, "pscomppars_info.csv"))
