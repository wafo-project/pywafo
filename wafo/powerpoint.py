'''
Created on 15. des. 2009

@author: pab
'''
# import os
# import sys
# import win32com
# from win32com.client.selecttlb import EnumTlbs
# typelib_mso = None
# typelib_msppt = None
# for typelib in EnumTlbs():
#    d = typelib.desc.split(' ')
#    if d[0] == 'Microsoft' and d[1] == 'Office' and d[3] == 'Object' \
#        and d[4] == 'Library':
#        typelib_mso = typelib
#    if d[0] == 'Microsoft' and d[1] == 'PowerPoint' and d[3] == 'Object' \
#            and d[4] == 'Library':
#        typelib_msppt = typelib
# if hasattr(sys, 'frozen'):  # If we're an .exe file
#    win32com.__gen_path__ = os.path.dirname(sys.executable)
# #    win32com.__gen_path__ = os.environ['TEMP']
# if win32com.client.gencache.is_readonly:
#    win32com.client.gencache.is_readonly = False
#    win32com.client.gencache.Rebuild()
# MSPPT = win32com.client.gencache.EnsureModule(typelib_msppt.clsid,
#                                typelib_msppt.lcid,
#                                int(typelib_msppt.major),
#                                int(typelib_msppt.minor))
# MSO = win32com.client.gencache.EnsureModule(typelib_mso.clsid,
#                        typelib_mso.lcid,
#                              int(typelib_mso.major), int(typelib_mso.minor))
from __future__ import absolute_import
from six import iteritems  # @UnresolvedImport
import os
import warnings
import win32com.client
from . import MSO
from . import MSPPT
from PIL import Image  # @UnresolvedImport

g = globals()
for c in dir(MSO.constants):
    g[c] = getattr(MSO.constants, c)
for c in dir(MSPPT.constants):
    g[c] = getattr(MSPPT.constants, c)


class Powerpoint(object):

    def __init__(self, file_name=''):

        self.application = win32com.client.Dispatch("Powerpoint.Application")
        # self.application.Visible = True
        self._visible = self.application.Visible
        if file_name:
            self.presentation = self.application.Presentations.Open(file_name)
        else:
            self.presentation = self.application.Presentations.Add()
        self.num_slides = 0
        # default picture width and height
        self.default_width = 500
        self.default_height = 400
        self.title_font = 'Arial'  # 'Boopee'
        self.title_size = 36
        self.text_font = 'Arial'  # 'Boopee'
        self.text_size = 20
        self.footer = ''

    def set_footer(self):
        '''
        Set Footer in SlideMaster and NotesMaster
        '''
        if self.footer:
            if self.presentation.HasTitleMaster:
                TMHF = self.presentation.TitleMaster.HeadersFooters
                TMHF.Footer.Text = self.footer
                TMHF.Footer.Visible = True

            SMHF = self.presentation.SlideMaster.HeadersFooters
            SMHF.Footer.Text = self.footer
            SMHF.Footer.Visible = True
            SMHF.SlideNumber.Visible = True
            NMHF = self.presentation.NotesMaster.HeadersFooters
            NMHF.Footer.Text = self.footer
            NMHF.SlideNumber.Visible = True
            for slide in self.presentation.Slides:
                shapes = slide.Shapes
                for shape in shapes:
                    if shape.Name == 'Footer':
                        footer = shape
                        break
                else:
                    footer = shapes.AddTextbox(
                        msoTextOrientationHorizontal,  # @UndefinedVariable
                        Left=0, Top=510, Width=720, Height=28.875)
                    footer.Name = 'Footer'
                footer.TextFrame.TextRange.Text = self.footer

    def add_title_slide(self, title, subtitle=''):
        self.num_slides += 1
        slide = self.presentation.Slides.Add(
            self.num_slides, MSPPT.constants.ppLayoutTitle)

        unused_title_id, unused_textbox_id = 1, 2
        for id_, title1 in enumerate([title, subtitle]):
            titlerange = slide.Shapes(id_ + 1).TextFrame.TextRange
            titlerange.Text = title1
            titlerange.Font.Name = self.title_font
            titlerange.Font.Size = self.title_size - id_ * \
                12 if self.title_size > 22 else self.title_size

    def add_slide(self, title='', texts='', notes='', image_file='',
                  maxlevel=None, left=220, width=-1, height=-1):
        self.num_slides += 1
        slide = self.presentation.Slides.Add(
            self.num_slides, MSPPT.constants.ppLayoutText)

        self.add2slide(slide, title, texts, notes, image_file, maxlevel, left,
                       width, height)
        return slide

    def add2slide(self, slide, title='', texts='', notes='', image_file='',
                  maxlevel=None, left=220, width=-1, height=-1,
                  keep_aspect=True):
        title_id, textbox_id = 1, 2
        if title:
            titlerange = slide.Shapes(title_id).TextFrame.TextRange
            titlerange.Font.Name = self.title_font
            titlerange.Text = title
            titlerange.Font.Size = self.title_size

        if texts != '' and texts != ['']:
            # textrange = slide.Shapes(textbox_id).TextFrame.TextRange
            self._add_text(slide, textbox_id, texts, maxlevel)

        if image_file != '' and image_file != ['']:
            if keep_aspect:
                im = Image.open(image_file)
                t_w, t_h = im.size
                if height <= 0 and width <= 0:
                    if t_w * self.default_height < t_h * self.default_width:
                        height = self.default_height
                    else:
                        width = self.default_width
                if height <= 0 and width:
                    height = t_h * width / t_w
                elif height and width <= 0:
                    width = t_w * height / t_h

            slide.Shapes.AddPicture(FileName=image_file, LinkToFile=False,
                                    SaveWithDocument=True,
                                    Left=left, Top=110,
                                    Width=width, Height=height)  # 400)
        if notes != '' and notes != ['']:
            notespage = slide.NotesPage  # .Shapes(2).TextFrame.TextRange
            self._add_text(notespage, 2, notes)
        return slide

    def _add_text(self, page, id, txt, maxlevel=None):  # @ReservedAssignment
        page.Shapes(id).TextFrame.TextRange.Font.Name = self.text_font

        if isinstance(txt, dict):
            self._add_text_from_dict(page, id, txt, 1, maxlevel)
        elif isinstance(txt, (list, tuple)):
            self._add_text_from_list(page, id, txt, maxlevel)
        else:
            unused_tr = page.Shapes(id).TextFrame.TextRange.InsertAfter(txt)
            unused_temp = page.Shapes(id).TextFrame.TextRange.InsertAfter('\r')

        page.Shapes(id).TextFrame.TextRange.Font.Size = self.text_size

    def _add_text_from_dict(self, page, id, txt_dict,  # @ReservedAssignment
                            level, maxlevel=None):
        if maxlevel is None or level <= maxlevel:
            for name, subdict in iteritems(txt_dict):
                tr = page.Shapes(id).TextFrame.TextRange.InsertAfter(name)
                unused_temp = page.Shapes(
                    id).TextFrame.TextRange.InsertAfter('\r')
                tr.IndentLevel = level
                self._add_text_from_dict(
                    page, id, subdict, min(level + 1, 5), maxlevel)

    def _add_text_from_list(self, page, id,  # @ReservedAssignment
                            txt_list, maxlevel=None):
        for txt in txt_list:
            level = 1
            while isinstance(txt, (list, tuple)):
                txt = txt[0]
                level += 1
            if maxlevel is None or level <= maxlevel:
                tr = page.Shapes(id).TextFrame.TextRange.InsertAfter(txt)
                unused_temp = page.Shapes(
                    id).TextFrame.TextRange.InsertAfter('\r')
                tr.IndentLevel = level

    def save(self, fullfile=''):
        if fullfile:
            self.presentation.SaveAs(FileName=fullfile)
        else:
            self.presentation.Save()

    def quit(self):  # @ReservedAssignment
        if self._visible:
            self.presentation.Close()
        else:
            self.application.Quit()

    def quit_only_if_hidden(self):
        if not self._visible:
            self.application.Quit()


def test_powerpoint():
    # Make powerpoint

    ppt = Powerpoint()
    ppt.footer = 'This is the footer'
    ppt.add_title_slide('Title', 'Per A.')
    ppt.add_slide(title='alsfkasldk', texts='asdflaf', notes='asdfas')
    ppt.set_footer()


def make_ppt():
    application = win32com.client.Dispatch("Powerpoint.Application")
    application.Visible = True
    presentation = application.Presentations.Add()
    slide1 = presentation.Slides.Add(1, MSPPT.constants.ppLayoutText)

#    title = slide1.Shapes.AddTextBox(Type=msoTextOrientationHorizontal,
#                Left=50, Top=10, Width=620, Height=70)
#    title.TextFrame.TextRange.Text = 'Overskrift'
    title_id, textbox_id = 1, 2
    slide1.Shapes(title_id).TextFrame.TextRange.Text = 'Overskrift'
    # slide1.Shapes(title_id).TextFrame.Width = 190

    slide1.Shapes(textbox_id).TextFrame.TextRange.InsertAfter('Test')
    unused_tr = slide1.Shapes(textbox_id).TextFrame.TextRange.InsertAfter('\r')
    slide1.Shapes(textbox_id).TextFrame.TextRange.IndentLevel = 1
    tr = slide1.Shapes(textbox_id).TextFrame.TextRange.InsertAfter('tests')
    unused_tr0 = slide1.Shapes(
        textbox_id).TextFrame.TextRange.InsertAfter('\r')
    tr.IndentLevel = 2
    tr1 = slide1.Shapes(textbox_id).TextFrame.TextRange.InsertAfter('test3')
    tr1.IndentLevel = 3
    # slide1.Shapes(textbox_id).TextFrame.TextRange.Text = 'Test \r test2'

#    textbox = slide1.Shapes.AddTextBox(Type=msoTextOrientationHorizontal,
#                    Left=30, Top=100, Width=190, Height=400)
#    textbox.TextFrame.TextRange.Text = 'Test \r test2'
    # picbox = slide1.Shapes(picb_id)

    filename = r'c:\temp\data1_report1_and_2_Tr120_1.png'
    slide1.Shapes.AddPicture(FileName=filename, LinkToFile=False,
                             SaveWithDocument=True,
                             Left=220, Top=100, Width=500, Height=420)

    slide1.NotesPage.Shapes(2).TextFrame.TextRange.Text = 'test'


#    for shape in slide1.Shapes:
#        shape.TextFrame.TextRange.Text = 'Test \r test2'
    # slide1.Shapes.Titles.TextFrames.TestRange.Text
#    shape = slide1.Shapes.AddShape(msoShapeRectangle, 300, 100, 400, 400)
#    shape.TextFrame.TextRange.Text = 'Test \n test2'
#    shape.TextFrame.TextRange.Font.Size = 12
    #
#    app = wx.PySimpleApp()
#    dialog = wx.FileDialog(None, 'Choose image file', defaultDir=os.getcwd(),
#                              wildcard='*.*',
#                              style=wx.OPEN | wx.CHANGE_DIR | wx.MULTIPLE)
#
#    if dialog.ShowModal() == wx.ID_OK:
#        files_or_paths = dialog.GetPaths()
#        for filename in files_or_paths:
#            slide1.Shapes.AddPicture(FileName=filename, LinkToFile=False,
#                                     SaveWithDocument=True,
#                                     Left=100, Top=100, Width=200, Height=200)
#    dialog.Destroy()
    # presentation.Save()
    # application.Quit()
def rename_ppt():
    root = r'C:/pab/tsm_opeval/analysis_tsmps_aco_v2008b/plots'
    filenames = os.listdir(root)
    prefix = 'TSMPSv2008b_'
    for filename in filenames:
        if filename.endswith('.ppt'):
            try:
                ppt = Powerpoint(os.path.join(root, filename))
                ppt.footer = prefix + filename
                ppt.set_footer()
                ppt.save(os.path.join(root, ppt.footer))
            except:
                warnings.warn('Unable to load %s' % filename)


def load_file_into_ppt():
    root = r'C:/pab/tsm_opeval/analysis_tsmps_aco_v2008b/plots'
    filenames = os.listdir(root)
    prefix = 'TSMPSv2008b_'
    for filename in filenames:
        if filename.startswith(prefix) and filename.endswith('.ppt'):
            try:
                unused_ppt = Powerpoint(os.path.join(root, filename))
            except:
                warnings.warn('Unable to load %s' % filename)
if __name__ == '__main__':
    # make_ppt()
    # test_powerpoint()
    # load_file_into_ppt()
    rename_ppt()
