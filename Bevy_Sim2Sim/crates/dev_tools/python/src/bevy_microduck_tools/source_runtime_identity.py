"""Actual installed-file identity; CPU collection never certifies lazy GPU coverage."""
from __future__ import annotations

import importlib
import importlib.metadata as metadata
import json
import os
import platform
import sys
from pathlib import Path

from .authorization import Rejection
from .serialization import identity, sha256_file

SCHEMA = 'microduck_source_runtime_identity_v1'
ROOT_DISTRIBUTIONS = ('mjlab','mujoco','mujoco-warp','warp-lang','torch','rsl-rl-lib',
                      'better-actuator-models','onnx','numpy','tensordict','onnxscript','onnx-ir','scipy')
CORE_MODULES = ('mjlab','mujoco','mujoco._structs','mujoco_warp','warp','bam','bam.mjlab',
                'rsl_rl','torch','torch._C','numpy','tensordict','onnx','onnxscript','onnx_ir')
BYTECODE = {'.pyc','.pyo'}


def _nonbytecode(path):
    locator=path.name=='RECORD' and path.parent.name.endswith('.dist-info')
    return '__pycache__' not in path.parts and path.suffix not in BYTECODE and not locator


def _files(roots, explicit):
    paths = set()
    for text in roots:
        root = Path(text)
        if not root.is_dir():
            raise Rejection(f'Runtime package root missing: {root}')
        for path in root.rglob('*'):
            if not _nonbytecode(path): continue
            if path.is_symlink() and path.is_dir():
                raise Rejection(f'Directory symlink requires explicit runtime coverage: {path}')
            if path.is_symlink() and not path.exists():
                raise Rejection(f'Broken runtime symlink: {path}')
            if path.is_file(): paths.add(os.path.abspath(path))
    for text in explicit:
        path = Path(text)
        if not path.is_file(): raise Rejection(f'Runtime input missing: {path}')
        if _nonbytecode(path): paths.add(os.path.abspath(path))
    return sorted(paths)


def file_record(path: Path) -> dict:
    try:
        first=path.stat(); digest=sha256_file(path); last=path.stat()
        if (first.st_dev,first.st_ino,first.st_size,first.st_mtime_ns) != (last.st_dev,last.st_ino,last.st_size,last.st_mtime_ns):
            raise Rejection(f'Runtime file changed during hashing: {path}')
        return {'path':os.path.abspath(path),'resolved_path':str(path.resolve()),'size':last.st_size,
                'sha256':digest,'symlink_target':os.readlink(path) if path.is_symlink() else None}
    except OSError as error:
        raise Rejection(f'Cannot read runtime file {path}: {error}') from error


def _python():
    return {'executable':os.path.abspath(sys.executable),'resolved_executable':str(Path(sys.executable).resolve()),
            'version':sys.version,'implementation':sys.implementation.name,'cache_tag':sys.implementation.cache_tag,
            'machine':platform.machine(),'prefix':sys.prefix,'base_prefix':sys.base_prefix}


def _resolution():
    return {name:os.environ.get(name) for name in ('LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH','PYTHONHOME')}


def _origins(modules):
    result=[]
    for name in modules:
        module=importlib.import_module(name)
        origin=getattr(module,'__file__',None)
        if origin is None: raise Rejection(f'Runtime module has no actual file origin: {name}')
        path=Path(origin)
        if path.suffix in BYTECODE:
            try: path=Path(importlib.util.source_from_cache(str(path)))
            except ValueError: raise Rejection(f'Unverifiable bytecode-only module: {name}') from None
        result.append({'module':name,'path':os.path.abspath(path),'resolved_path':str(path.resolve()),
                       'loader':type(module.__loader__).__name__})
    return result


def _native_maps():
    paths=set()
    try: lines=Path('/proc/self/maps').read_text().splitlines()
    except OSError as error: raise Rejection(f'Cannot inspect actual native mappings: {error}') from error
    for line in lines:
        parts=line.split(maxsplit=5)
        if len(parts)!=6 or not parts[5].startswith('/') or '.so' not in Path(parts[5]).name: continue
        if parts[5].endswith(' (deleted)'): raise Rejection('Deleted mapped native library cannot be identified')
        path=Path(parts[5])
        try:
            stat=path.stat();major,minor=(int(value,16) for value in parts[3].split(':'))
        except (OSError,ValueError) as error:raise Rejection(f'Cannot identify mapped native library {path}: {error}') from error
        if (stat.st_ino,os.major(stat.st_dev),os.minor(stat.st_dev)) != (int(parts[4]),major,minor):
            raise Rejection(f'Mapped native device/inode no longer matches disk: {path}')
        paths.add(os.path.abspath(path))
    return sorted(paths)


def _distribution_closure(roots):
    # Packaging is supplied by the adopted scientific environment, not by the game.
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    pending=[(name,set()) for name in roots]; seen={}; rows={}; missing=[]
    while pending:
        name,extras=pending.pop(0); name=canonicalize_name(name)
        before=seen.get(name)
        if before is not None and extras <= before: continue
        seen[name]=(before or set())|extras
        try: dist=metadata.distribution(name)
        except metadata.PackageNotFoundError:
            missing.append(f'{name}: not installed');continue
        if dist.files is None:
            missing.append(f'{name}: no installed file locator');continue
        root_paths=set();explicit=set()
        for entry in dist.files:
            parts=entry.parts
            if not parts or not _nonbytecode(entry):continue
            path=Path(os.path.abspath(dist.locate_file(entry)))
            if not path.exists():
                missing.append(f'{name}: missing located file {path}');continue
            if '..' in parts or len(parts)==1:
                if path.is_file():explicit.add(str(path))
                continue
            # Shared namespaces must not repeatedly scan unrelated vendor packages.
            width=2 if parts[0] in {'nvidia','google','backports'} else 1
            prefix=Path(os.path.abspath(dist.locate_file(Path(*parts[:width]))))
            if prefix.is_dir():root_paths.add(str(prefix))
            elif path.is_file():explicit.add(str(path))
        metadata_path=Path(os.path.abspath(dist.locate_file('')))
        for requirement_text in dist.requires or []:
            requirement=Requirement(requirement_text)
            contexts=[{'extra':''},*({'extra':extra} for extra in seen[name])]
            if requirement.marker is not None and not any(requirement.marker.evaluate(context) for context in contexts):continue
            try: installed=metadata.version(requirement.name)
            except metadata.PackageNotFoundError:
                missing.append(f'{name} requires {requirement}: not installed');continue
            if requirement.specifier and not requirement.specifier.contains(installed,prereleases=True):
                missing.append(f'{name} requires {requirement}: installed {installed}')
            pending.append((requirement.name,set(requirement.extras)))
        rows[name]={'name':name,'version':dist.version,'installation_root':str(metadata_path),
                    'package_roots':sorted(root_paths),'explicit_files':sorted(explicit),'requirements':dist.requires or []}
    return [rows[name] for name in sorted(rows)],sorted(set(missing))


def collect_source_runtime(source_root: Path, *, roots=ROOT_DISTRIBUTIONS, modules=CORE_MODULES) -> dict:
    """Complete installed-byte collection on CPU; lazy/native profile remains unverified."""
    source_root=source_root.resolve()
    source_path=str(source_root/'src')
    if source_path not in sys.path:sys.path.insert(0,source_path)
    import mjlab_microduck.tasks  # Official import order; no environment/robot construction.
    if not Path(mjlab_microduck.tasks.__file__).resolve().is_relative_to(source_root):
        raise Rejection('Wrong actual imported upstream source')
    distributions,missing=_distribution_closure(roots)
    imports=_origins(modules)
    native=_native_maps()
    package_roots=sorted({path for row in distributions for path in row['package_roots']})
    explicit=sorted({path for row in distributions for path in row['explicit_files']}|
                    {os.path.abspath(sys.executable),*native})
    files=_files(package_roots,explicit)
    records=[file_record(Path(path)) for path in files]
    indexed={row['path']:row for row in records}
    origins_complete=all(row['path'] in indexed for row in imports)
    coverage={'installed_file_sets_complete':not missing,'recursive_dependencies_complete':not missing,
              'cpu_import_origins_complete':origins_complete,'native_profile_complete':False,'complete':False,
              'missing_dependencies':missing,
              'limitations':['CPU imports/maps only; full source lazy/native/JIT profile not executed or verified',
                             'Trusted filesystem preflight; shared scientific environment must not be concurrently modified']}
    payload={'schema':SCHEMA,'source_root':str(source_root),'required_distributions':list(roots),
             'python':_python(),'resolution':_resolution(),'distributions':distributions,
             'package_roots':package_roots,'explicit_files':explicit,'files':records,'imports':imports,
             'native_libraries':native,'coverage':coverage}
    payload['identity']=identity(payload)
    return payload


def _exact(value, keys, label):
    if not isinstance(value,dict) or set(value)!=set(keys):raise Rejection(f'Unknown or missing runtime {label} fields')


def _strings(values, label, *, absolute=False, ordered=False, nonempty=False):
    if (not isinstance(values,list) or (nonempty and not values) or
            any(not isinstance(value,str) or not value for value in values)):
        raise Rejection(f'Invalid runtime {label}')
    if absolute and any(not Path(value).is_absolute() for value in values):
        raise Rejection(f'Runtime {label} must contain absolute paths')
    if ordered and values!=sorted(set(values)):
        raise Rejection(f'Duplicate or unordered runtime {label}')


def validate_receipt(receipt: dict) -> None:
    _exact(receipt,{'schema','source_root','required_distributions','python','resolution','distributions',
                    'package_roots','explicit_files','files','imports','native_libraries','coverage','identity'},'top-level')
    if receipt['schema']!=SCHEMA:raise Rejection('Unknown source runtime identity schema')
    if not isinstance(receipt['source_root'],str) or not Path(receipt['source_root']).is_absolute():raise Rejection('Invalid runtime source root')
    if identity({key:value for key,value in receipt.items() if key!='identity'})!=receipt['identity']:
        raise Rejection('Source runtime receipt payload changed')
    _exact(receipt['python'],{'executable','resolved_executable','version','implementation','cache_tag','machine','prefix','base_prefix'},'python')
    if any(not isinstance(value,str) or not value for value in receipt['python'].values()):raise Rejection('Invalid actual Python identity')
    _exact(receipt['resolution'],{'LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH','PYTHONHOME'},'resolution')
    if any(value is not None and not isinstance(value,str) for value in receipt['resolution'].values()):raise Rejection('Invalid runtime resolution')
    coverage=receipt['coverage']
    _exact(coverage,{'installed_file_sets_complete','recursive_dependencies_complete','cpu_import_origins_complete',
                     'native_profile_complete','complete','missing_dependencies','limitations'},'coverage')
    flags=('installed_file_sets_complete','recursive_dependencies_complete','cpu_import_origins_complete','native_profile_complete','complete')
    if any(type(coverage[key]) is not bool for key in flags):raise Rejection('Runtime coverage flags must be boolean')
    _strings(coverage['missing_dependencies'],'missing dependencies')
    _strings(coverage['limitations'],'coverage limitations')
    if coverage['complete'] != (all(coverage[key] for key in flags[:-1]) and not coverage['missing_dependencies'] and not coverage['limitations']):
        raise Rejection('Runtime complete flag contradicts actual coverage')
    for key in ('package_roots','explicit_files','native_libraries'):
        _strings(receipt[key],key,absolute=True,ordered=True)
    _strings(receipt['required_distributions'],'scientific roots',nonempty=True)
    if len(set(receipt['required_distributions']))!=len(receipt['required_distributions']):raise Rejection('Duplicate scientific roots')
    if not isinstance(receipt['distributions'],list) or not receipt['distributions']:raise Rejection('Runtime distribution coverage empty')
    for row in receipt['distributions']:
        _exact(row,{'name','version','installation_root','package_roots','explicit_files','requirements'},'distribution')
        if not all(isinstance(row[key],str) and row[key] for key in ('name','version','installation_root')):raise Rejection('Invalid distribution identity')
        for key in ('package_roots','explicit_files'):
            _strings(row[key],'distribution '+key,absolute=True,ordered=True)
        if not isinstance(row['requirements'],list) or any(not isinstance(value,str) for value in row['requirements']):
            raise Rejection('Invalid distribution dependency records')
    if receipt['package_roots']!=sorted({path for row in receipt['distributions'] for path in row['package_roots']}):
        raise Rejection('Runtime package root aggregate is incomplete')
    names=[row['name'] for row in receipt['distributions']]
    if names!=sorted(set(names)):raise Rejection('Duplicate or unordered runtime distributions')
    if coverage['recursive_dependencies_complete'] and not set(receipt['required_distributions'])<=set(names):raise Rejection('Missing declared scientific root distributions')
    required_explicit={path for row in receipt['distributions'] for path in row['explicit_files']}|set(receipt['native_libraries'])|{receipt['python']['executable']}
    if receipt['explicit_files']!=sorted(required_explicit):raise Rejection('Runtime explicit file aggregate is incomplete')
    if not isinstance(receipt['files'],list) or not receipt['files']:raise Rejection('Runtime actual file coverage empty')
    paths=[]
    for row in receipt['files']:
        _exact(row,{'path','resolved_path','size','sha256','symlink_target'},'file')
        if not all(isinstance(row[key],str) and Path(row[key]).is_absolute() for key in ('path','resolved_path')):raise Rejection('Invalid runtime file origin')
        if type(row['size']) is not int or row['size']<0:raise Rejection('Invalid runtime file size')
        digest=row['sha256']
        if not isinstance(digest,str) or len(digest)!=64 or any(char not in '0123456789abcdef' for char in digest):raise Rejection('Invalid actual runtime SHA256')
        if row['symlink_target'] is not None and not isinstance(row['symlink_target'],str):raise Rejection('Invalid runtime symlink')
        paths.append(row['path'])
    if paths!=sorted(set(paths)):raise Rejection('Duplicate or unordered runtime files')
    if not isinstance(receipt['imports'],list):raise Rejection('Runtime import origins must be a list')
    modules=[]
    for row in receipt['imports']:
        _exact(row,{'module','path','resolved_path','loader'},'import')
        if not all(isinstance(row[key],str) and row[key] for key in ('module','path','resolved_path','loader')):raise Rejection('Invalid actual import identity')
        modules.append(row['module'])
        if row['path'] not in paths:raise Rejection('Imported scientific file absent from actual-byte catalogue')
    if len(modules)!=len(set(modules)):raise Rejection('Duplicate actual import identity')
    for path in receipt['native_libraries']:
        if path not in paths:raise Rejection('Native library absent from actual-byte catalogue')
    if receipt['python']['executable'] not in paths:raise Rejection('Actual Python executable absent from catalogue')


def verify_source_runtime(receipt: dict, *, source_root: Path, require_complete: bool=True) -> dict:
    """Re-enumerate file sets and read bytes; no RECORD or stat-only identity shortcut."""
    validate_receipt(receipt)
    if require_complete and not receipt['coverage']['complete']:
        raise Rejection('Source runtime identity is incomplete; no grandfathered learning')
    if str(source_root.resolve())!=receipt['source_root']:raise Rejection('Source runtime root changed')
    if _python()!=receipt['python'] or _resolution()!=receipt['resolution']:
        raise Rejection('Actual Python or runtime import/library resolution changed')
    distributions,missing=_distribution_closure(receipt['required_distributions'])
    if distributions!=receipt['distributions'] or missing!=receipt['coverage']['missing_dependencies']:
        raise Rejection('Runtime installed version/location/dependency/file-locator closure changed')
    paths=_files(receipt['package_roots'],receipt['explicit_files'])
    if paths!=[row['path'] for row in receipt['files']]:raise Rejection('Runtime actual nonbytecode file set changed')
    for row in receipt['files']:
        if file_record(Path(row['path']))!=row:raise Rejection(f'Runtime actual bytes/origin changed: {row["path"]}')
    if any(row['module'] in {'mjlab','bam','bam.mjlab'} for row in receipt['imports']):
        source_path=str(source_root.resolve()/'src')
        if source_path not in sys.path:sys.path.insert(0,source_path)
        import mjlab_microduck.tasks
        if not Path(mjlab_microduck.tasks.__file__).resolve().is_relative_to(source_root.resolve()):
            raise Rejection('Actual scientific upstream import root changed')
    if _origins([row['module'] for row in receipt['imports']])!=receipt['imports']:
        raise Rejection('Actual loaded scientific import origin changed')
    mapped=_native_maps()
    if any(path not in paths for path in mapped):
        raise Rejection('Actual native loader used an unbound library')
    return {'identity':receipt['identity'],'file_count':len(paths),'bytes':sum(row['size'] for row in receipt['files']),
            'complete':receipt['coverage']['complete'],'verification':'actual complete declared file sets and bytes, metadata versions, import and mapped origins'}


def verify_candidate_runtime(candidate: dict, source_root: Path) -> dict:
    artifact=candidate.get('artifacts',{}).get('source_runtime_identity')
    if artifact is None:raise Rejection('Candidate lacks complete source runtime identity; historical candidates cannot learn')
    if (not isinstance(artifact,dict) or not isinstance(artifact.get('path'),str) or
            not isinstance(artifact.get('sha256'),str)):
        raise Rejection('Invalid candidate source runtime identity artifact')
    path=Path(artifact['path'])
    if not path.is_file() or sha256_file(path)!=artifact['sha256']:
        raise Rejection('Candidate source runtime identity artifact changed or missing')
    try:receipt=json.loads(path.read_text())
    except (ValueError,OSError) as error:raise Rejection(f'Cannot read source runtime receipt: {error}') from error
    validate_receipt(receipt)
    if not set(ROOT_DISTRIBUTIONS) <= set(receipt.get('required_distributions',[])):
        raise Rejection('Source runtime receipt omits required scientific distributions')
    if not set(CORE_MODULES) <= {row.get('module') for row in receipt.get('imports',[])}:
        raise Rejection('Source runtime receipt omits required scientific import origins')
    return verify_source_runtime(receipt,source_root=source_root,require_complete=True)
