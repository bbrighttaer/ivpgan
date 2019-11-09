import os
import xml.etree.ElementTree as et


def create_bash_script(configuration):
    py_file = args = None
    for child in configuration:
        if len(child.attrib) > 0 and "name" in child.attrib:
            if "SCRIPT_NAME" == child.attrib["name"]:
                py_file = child.attrib["value"].split('/')[-1]
            if "PARAMETERS" == child.attrib["name"]:
                args = child.attrib["value"]
    if py_file is not None and args is not None:
        cmd = 'python {} {}'.format(py_file, args)
        script_file = 'bash_files/{}.sh'.format(configuration.attrib["name"])
        with open(script_file, 'w') as script:
            script.write(cmd)
            print("{} created successfully!".format(script_file))


def create_scripts(files):
    for file in files:
        cf = et.parse(file).getroot().find('configuration')
        create_bash_script(cf)


if __name__ == '__main__':
    files = os.listdir('./')
    files = list(filter(lambda x: '.xml' in x, files))
    print("Number of files=%d" % len(files))
    if not os.path.exists("bash_files"):
        os.mkdir("bash_files")
    create_scripts(files)
