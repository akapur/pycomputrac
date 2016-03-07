import os
import fnmatch
import struct
import datetime
import numpy as np
from pandas import DataFrame
import pandas as pd


def strip_null(c_string, null=b'\x00'):
    """Strip everything past the first null in a string

    Useful if a null terminated string is read from a binary format file
    with a fixed width string field.  "null" may be any character (or string). 
    The character with signals end of string is not part of the string returned. 
    @param c_string: the string to clean up.
    @param null: A sentinel character (or string) used to signal end of string.
    """
    end_idx = c_string.find(null)
    return c_string[0:end_idx]


def date2string(dt: datetime.date) -> str:
    """Returns a string YYYY-MM-DD given a datetime.date object

    @param dt: a datetime.date object
    """
    return "%0.4d-%0.2d-%0.2d" % (dt.year, dt.month, dt.day)


def fmsfloat2date(ms_date: float) -> datetime.date:
    """Convert a metastock format date float into a datetime.date

    Metastock stores dates as a 4 byte float.  This function returns a
    datetime.date object given a metastock 4 byte float date
    @param ms_date: a float representing a metastock date.
    """
    yyyymmdd = int(ms_date+19000000)
    yyyy = int(yyyymmdd/10000)
    mm = int((yyyymmdd - (yyyy*10000))/100)
    dd = int(yyyymmdd % 100)
    if (1900 == yyyy) and (0 == mm) and (0 == dd):
        return datetime.date(1900, 1, 1)
    else:
        return datetime.date(yyyy, mm, dd)


def fmsfloat2datetime(ms_date) -> datetime.date:
    """Convert a metastock format date float into a datetime.datetime

    Metastock stores dates as a 4 byte float.  This function returns a
    datetime.date object given a metastock 4 byte float date
    @param ms_date: a float representing a metastock date.
    """
    yyyymmdd = int(ms_date+19000000)
    yyyy = yyyymmdd/10000
    mm = (yyyymmdd - (yyyy*10000))/100
    dd = yyyymmdd % 100
    if (1900 == yyyy) and (0 == mm) and (0 == dd):
        return datetime.date(1900, 1, 1)
    else:
        return datetime.date(yyyy, mm, dd)


def fmsbin2ieee(ms_bin) -> float:

    """Convert an MS Basic format float to a IEEE format float

    Given a 4 byte buffer containing a MS Basic format float return a 4 byte
    IEEE float (i.e. numpy dtype of float32
    @param ms_bin: A byte buffer containing the Microsoft Basic format float used by Metastock

    MS Binary Format
    byte order =>    m3 | m2 | m1 | exponent
    m1 is most significant byte => sbbb|bbbb
    m3 is the least significant byte
           m = mantissa byte
           s = sign bit
           b = bit

    IEEE Single Precision Float Format
       m3        m2        m1     exponent
    mmmm|mmmm mmmm|mmmm emmm|mmmm seee|eeee
             s = sign bit
             e = exponent bit
             m = mantissa bit

    MBF is bias 128 and IEEE is bias 127. ALSO, MBF places
    the decimal point before the assumed bit, while
    IEEE places the decimal point after the assumed bit.

   """

    # Any ms_binary with 0 exponent is 0
    if 0 == ms_bin[3]:
        return float(0)

    ieee = bytearray(4)

    # 1000|0000b
    sign = ms_bin[2] & 0x80
    ieee[3] |= (sign % 256)

    # Simplified from ms_bin[3]-1-128+127
    ieee_exp = ms_bin[3] - 2

    # The first 7 bits of the exponent in ieee[3]
    ieee[3] |= ((ieee_exp >> 1) % 256)

    # The 1 remaining bit in the first bin of ieee[2]
    ieee[2] |= ((ieee_exp << 7) % 256)

    # 0111|1111b : mask out the ms_bin sign bit
    ieee[2] |= ((ms_bin[2] & 0x7f) % 256)
    ieee[1] = ms_bin[1]
    ieee[0] = ms_bin[0]

    return struct.unpack("f", ieee)[0]


class ComputracDir(object):
    """Provides easy access to historical market data in Metastock Format.

    Abstracts away stuff necessary to read the data.  Needs an index in emaster
    format.  Reads the index and buffers it.  Reads data for each security on
    demand and returns it.  Does not buffer each security's data since this
    could in theory exceed memory available.

    This class as specifically used to read data as distributed by Norgate 
    (also called PremiumData).  This is a specific variant of the Computrac 
    format.  This class is also designed to return data a numpy/pandas friendly
    format as opposed to generate csv files for export.

    Metastock is a program for handling financial information.  Prior to 2012
    it could read or provide data in a format initially created by Computrac.
    However in 2012 it can only read data provided by Reuters/Datalink.  The
    Computrac/Metastock format is used by many data analysis programs and data
    providers.
    """

    def __init__(self, root_dir='', emaster_name='emaster'):
        self.num_files = 0
        self.max_file_num = 0

        self._ticker_refdata = {}
        self._name_refdata = {}
        self._emaster_files = []

        self.reset_refdata()
        if '' != root_dir:
            self.open_base_directory(root_dir, emaster_name)

    def __str__(self) -> str:
        return self.tickers.__str__()

    def reset_refdata(self) -> None:
        """Remove all data read from emaster files"""
        self._ticker_refdata = {}
        self._name_refdata = {}
        self._emaster_files = [] 

    def find_emaster_files(self, root_dir, emaster_name='emaster'):
        """Search the current directory and subdirs for all emaster files

        @param root_dir: FQ name of the root directory
        @param emaster_name: Name of emaster files, usually emaster
        """
        if not os.path.isdir(root_dir):
            raise Exception("Directory %s does not exist" % root_dir)

        matches = []
        for root, dirnames, filenames in os.walk(root_dir):
            for filename in fnmatch.filter(filenames, emaster_name):
                matches.append(os.path.join(root, filename))
        return matches

    def read_emaster_file(self, emaster_name):
        """Open and read the emaster file and cache data in it

        @param emaster_name: FQ Name of the emaster format file
        """

        if not os.path.isfile(emaster_name):
            raise Exception("File %s does not exist." % emaster_name)

        if emaster_name in self._emaster_files:
            raise Exception("%s has already been read" % emaster_name)

        header_fmt = "H H 188x"
        """ The format string for package struct to read the emaster header

        There is a single line header in the emaster file.  The format is:
        totalFiles: USHORT = Num files listed in the emaster file
        file_num: USHORT = Last (highest) file number mentioned in file.
        188 blank spaces
        """

        record_fmt = "2x B 3x B 2x c x 21s 28s c 3x f 4x f 63x 53s "
        """ The format string for package struct to read each emaster record

        There are many records in each emaster file each with format
        2 blanks often "30"
        fileFNumber: uchar = The FNumber of the file with the data.  Filename
            is F<FNumber>
        3 blanks
        totalFields: uchar = Number of 4 byte data fields in each record of the
            DATA file named F<FNumber>
        2 blanks
        flag: char = Either ' ' or '*' for autorun.  ????
        1 blank
        symbol: char[21] = NULL terminated string with symbol
        name: char[28] = NULL terminated string with asset name or description
        periodicity: char = Periodicity of data: 'D', 'W', 'M', etc
        3 blanks
        firstDate: float = First date for which there is data in metastock fmt.
        4 blanks
        lastDate: float = Last date for which there is data
        63 blanks: These blanks may be used for custom data
        longName: char[53] = Full name is placed here if name field was truncated
        """

        header_len = struct.calcsize(header_fmt)
        record_len = struct.calcsize(record_fmt)
        assert (header_len == record_len)

        with open(emaster_name, 'rb') as emaster:
            buf = emaster.read(header_len)
            self.num_files, self.max_file_num = struct.unpack(header_fmt, buf)
            buf = emaster.read(record_len)
            while len(buf) > 0:
                assert (len(buf) == record_len)
                f_num, num_fld, flag, symbol, name, freq, first_dt, last_dt, full_name = struct.unpack(record_fmt, buf)
                filename = os.path.join(os.path.dirname(emaster_name), 'F%d.dat' % f_num)
                flag = flag.decode()
                symbol = strip_null(symbol).decode()
                name = strip_null(name).decode()
                freq = freq.decode()
                first_dt = fmsfloat2date(first_dt)
                last_dt = fmsfloat2date(last_dt)
                if full_name[0] != 0:
                    name = strip_null(full_name).decode()
                record = (symbol, name, first_dt, last_dt, freq, filename, num_fld,
                          flag, emaster_name)
                if symbol in self._ticker_refdata:
                    old_record = self._ticker_refdata[symbol]
                    raise Exception("Duplicate ticker from dir %s, new dir %s" % (old_record[5], filename))
                if name in self._name_refdata:
                    old_record = self._name_refdata[name]
                    raise Exception("Duplicate name from dir %s, new dir %s" % (old_record[5], filename))
                self._ticker_refdata[symbol] = record
                self._name_refdata[name] = record
                buf = emaster.read(record_len)
            self._emaster_files.append(emaster_name)

    def open_base_directory(self, root_dir, emaster_name='emaster'):
        """Find and open all emaster files in the root dir and subdirs

        By calling this function on the root data directory you should be able
        to access all data in various metastock files in the directory and all
        subdirectories of the root data directory
        @param root_dir: The root directory which contains all Computrac data
        @param emaster_name: The name of emaster files
        """
        emaster_names = self.find_emaster_files(root_dir, emaster_name)
        for emaster_name in emaster_names:
            self.read_emaster_file(emaster_name)

    @property
    def tickers(self):
        """Return all tickers for which we have data"""
        return np.sort(list(self._ticker_refdata.keys()))

    @property
    def names(self):
        """Return all asset names for which we have data"""
        return np.sort(list(self._name_refdata.keys()))

    @property
    def emaster_files(self):
        return self._emaster_files

    def get_reference_data(self, asset_id):
        """Get the reference data for an asset given it's name or ticker
        @param asset_id: May be a ticker or asset name
        """
        if asset_id in self._ticker_refdata:
            return self._ticker_refdata[asset_id]
        elif asset_id in self._name_refdata:
            return self._name_refdata[asset_id]
        else:
            raise LookupError("Bad asset id %s" % asset_id)

    def get_raw_data(self, asset_id):
        """Return the underlying data given the name or ticker of an asset
        @param asset_id: may be a ticker or an asset name
        """
        header_fmt = "H H 24x"
        """Format string for the header of a 7 field data file
        totalRecords: ushort = Not used by metastock itself
        lastRecord: ushort = The last record in the file
        24 blanks
        """
        record_fmt = "f f f f f f f"
        """Format string for a 7 field metastock data record
        All floats, no padding, but the floats are in MS Basic format not
        IEEE format so you can't use struct unpack with the record_fmt as a
        format string.  Instead use the function fmsbin2ieee with the relevant
        4 raw bytes as an argument
        date, open, high, low, close, volume, openInterest
        """

        header_len = struct.calcsize(header_fmt)
        record_len = struct.calcsize(record_fmt)
        assert(header_len == record_len)
        (symbol, name, first_dt, last_dt, freq, file_name, num_fld, flag, master_file) =\
            self.get_reference_data(asset_id)
        with open(file_name, 'rb') as datafile:
            buf = datafile.read(header_len)
            (junk, num_records) = struct.unpack(header_fmt, buf)
            ohlc_data = np.empty(shape=(num_records-1),
                                 dtype=[('date',          'S10'),
                                        ('open',          'float'),
                                        ('high',          'float'),
                                        ('low',           'float'),
                                        ('close',         'float'),
                                        ('volume',        'float'),
                                        ('open_interest', 'float')])
            rec_num = 0
            buf = datafile.read(record_len)
            while len(buf) > 0:
                assert(len(buf) == record_len)
                ohlc_data[rec_num] = (
                    date2string(fmsfloat2date(fmsbin2ieee(buf[0:4]))),
                    fmsbin2ieee(buf[4:8]),
                    fmsbin2ieee(buf[8:12]),
                    fmsbin2ieee(buf[12:16]),
                    fmsbin2ieee(buf[16:20]),
                    fmsbin2ieee(buf[20:24]),
                    fmsbin2ieee(buf[24:28]))
                rec_num += 1
                buf = datafile.read(record_len)

        return ohlc_data

    def __getitem__(self, asset_id):
        return self.get_dataframe(asset_id)

    def get_dataframe(self, asset_id):
        """Return a pandas dataframe with price data for an asset 
        @param asset_id: may be a ticker or an asset name
        """
 
        raw_data = self.get_raw_data(asset_id)
        df = DataFrame(data={'open':     raw_data['open'],
                             'high':     raw_data['high'],
                             'low':      raw_data['low'],
                             'close':    raw_data['close'],
                             'volume':   raw_data['volume'],
                             'open_int': raw_data['open_interest']},
                       columns=['open', 'high', 'low', 'close', 'volume', 'open_int'],
                       index=pd.DatetimeIndex(raw_data['date']))
        return df







