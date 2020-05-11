import os

_model_classes = []
_table_schemas = {}


def _init_data(app_name):
    from django.apps import apps
    app = apps.get_app_config(app_name)
    models = app.get_models()
    for model in models:
        model_class = model.__name__
        _model_classes.append(model_class)
        fields = model._meta.get_fields()
        _table_schemas[model_class] = []

        for field in fields:
            _table_schemas[model_class].append(field.name)


def _create_directory(table_name, root_dir='webapi'):
    from django.conf import settings
    webapi_dir = os.path.join(settings.REPOSITORY_ROOT, root_dir)
    directory_name1 = table_name.split('_')[0]
    directory_name = os.path.join(webapi_dir, directory_name1)

    try:
        os.stat(directory_name)
    # noinspection PyBroadException
    except:
        os.mkdir(directory_name)

    return directory_name


def _create_package_init(app):
    dir_name = _create_directory(app)
    file_name = dir_name + "/" + "__init__.py"
    file = open(file_name, "w")
    file.close()


def _create_urls(app):
    content = 'from .routers import router\n'
    content += 'urlpatterns = router.urls\n'
    dir_name = _create_directory(app)
    file_name = dir_name + "/" + "urls.py"
    file = open(file_name, "w")
    file.writelines(content)
    file.close()


def _replace_underscore_with_hyphen(input_str):
    return input_str.replace('_', '-')


def _remove_underscore(input_str):
    return input_str.replace('_', '')


def _create_routes(app):
    content = 'from rest_framework.routers import DefaultRouter\n'
    dir_name = _create_directory(app)
    file_name = dir_name + "/" + "routers.py"
    # generate from .views
    content += 'from .views import (\n'
    for model_class in _model_classes:
        content += '    {}ViewSet,\n'.format(_remove_underscore(model_class))

    content += ')\n'
    content += "\nrouter = DefaultRouter()\n"

    # generate route register file
    for model_class in _model_classes:
        content += "router.register(r'{}s', {}ViewSet, base_name='{}s')\n".format(
            _replace_underscore_with_hyphen(model_class.lower()),
            _remove_underscore(model_class),
            _replace_underscore_with_hyphen(model_class.lower()))

    content += "\n"

    file = open(file_name, "w")
    file.writelines(content)
    file.close()


def _create_serializers(app):
    content = 'from rest_framework.serializers import ModelSerializer\n\n'
    dir_name = _create_directory(app)
    file_name = dir_name + "/" + "serializers.py"
    # generate from .models
    content += 'from {}.models import (\n'.format(app)
    for model_class in _model_classes:
        content += '    {},\n'.format(model_class)

    content += ')\n\n\n'

    # generate serializer classes
    for model_class in _model_classes:
        content += "class {}Serializer(ModelSerializer):\n\n".format(_remove_underscore(model_class))
        content += "    class Meta:\n"
        content += "        model = {}\n".format(model_class)
        content += "        fields = [\n"
        for column_name in _table_schemas[model_class]:
            content += "            '{}',\n".format(column_name)
        content += "        ]\n\n\n"

    content += "\n"

    file = open(file_name, "w")
    file.writelines(content)
    file.close()


def _create_views(app):
    content = 'from rest_framework.viewsets import ModelViewSet\n\n'
    dir_name = _create_directory(app)
    file_name = dir_name + "/" + "views.py"
    # generate from .models

    content += 'from {}.models import (\n'.format(app)
    for model_class in _model_classes:
        content += '    {},\n'.format(model_class)

    content += ')\n\n\n'

    # generate from .serializers

    content += 'from .serializers import (\n'
    for model_class in _model_classes:
        content += '    {}Serializer,\n'.format(_remove_underscore(model_class))

    content += ')\n\n\n'

    # generate view set classes
    for model_class in _model_classes:
        content += "class {}ViewSet(ModelViewSet):\n".format(_remove_underscore(model_class))
        content += "    queryset = {}.objects.all()\n".format(model_class)
        content += "    serializer_class = {}Serializer\n\n\n".format(_remove_underscore(model_class))

    content += "\n"

    file = open(file_name, "w")
    file.writelines(content)
    file.close()


def _create_model(app, model_class):
    model_name = model_class.lower()
    entity_path = app + '/' + model_name
    dir_name = _create_directory(entity_path, 'typescripts')
    content = "export interface {} {{\r\n".format(model_class)

    file_name = dir_name + "/" + "{}.model.ts".format(model_name)
    table_schema = _table_schemas[model_class]
    for field in table_schema:
        content += "  {}: string;\r\n".format(field)

    content += "}\r\n"

    file = open(file_name, "w")
    file.writelines(content)
    file.close()


def _create_models(app):
    for model_class in _model_classes:
        _create_model(app, model_class)


def _create_action(app, model_class):
    model_name = model_class.lower()
    model_name_upper = model_class.upper()
    entity_path = app + '/' + model_name
    dir_name = _create_directory(entity_path, 'typescripts')

    templateString = r'''import { Action } from '@ngrx/store';
import { Update, Predicate, EntityMap } from '@ngrx/entity';

import { Xxxx } from './xxxx.model';

export enum XxxxActionTypes {
  LOAD_XXXXS = '[Xxxx] Load Xxxxs',
  ADD_XXXX = '[Xxxx] Add Xxxx',
  UPSERT_XXXX = '[Xxxx] Upsert Xxxx',
  ADD_XXXXS = '[Xxxx] Add Xxxxs',
  UPSERT_XXXXS = '[Xxxx] Upsert Xxxxs',
  UPDATE_XXXX = '[Xxxx] Update Xxxx',
  UPDATE_XXXXS = '[Xxxx] Update Xxxxs',
  MAP_XXXXS = '[Xxxx] Map Xxxxs',
  DELETE_XXXX = '[Xxxx] Delete Xxxx',
  DELETE_XXXXS = '[Xxxx] Delete Xxxxs',
  DELETE_XXXXS_BY_PREDICATE = '[Xxxx] Delete Xxxxs By Predicate',
  CLEAR_XXXXS = '[Xxxx] Clear Xxxxs',
}

export class LoadXxxxs implements Action {
  readonly type = XxxxActionTypes.LOAD_XXXXS;

  constructor(public payload: { xxxxs: Xxxx[] }) {}
}

export class AddXxxx implements Action {
  readonly type = XxxxActionTypes.ADD_XXXX;

  constructor(public payload: { xxxx: Xxxx }) {}
}

export class UpsertXxxx implements Action {
  readonly type = XxxxActionTypes.UPSERT_XXXX;

  constructor(public payload: { xxxx: Xxxx }) {}
}

export class AddXxxxs implements Action {
  readonly type = XxxxActionTypes.ADD_XXXXS;

  constructor(public payload: { xxxxs: Xxxx[] }) {}
}

export class UpsertXxxxs implements Action {
  readonly type = XxxxActionTypes.UPSERT_XXXXS;

  constructor(public payload: { xxxxs: Xxxx[] }) {}
}

export class UpdateXxxx implements Action {
  readonly type = XxxxActionTypes.UPDATE_XXXX;

  constructor(public payload: { xxxx: Update<Xxxx> }) {}
}

export class UpdateXxxxs implements Action {
  readonly type = XxxxActionTypes.UPDATE_XXXXS;

  constructor(public payload: { xxxxs: Update<Xxxx>[] }) {}
}

export class MapXxxxs implements Action {
  readonly type = XxxxActionTypes.MAP_XXXXS;

  constructor(public payload: { entityMap: EntityMap<Xxxx> }) {}
}

export class DeleteXxxx implements Action {
  readonly type = XxxxActionTypes.DELETE_XXXX;

  constructor(public payload: { id: string }) {}
}

export class DeleteXxxxs implements Action {
  readonly type = XxxxActionTypes.DELETE_XXXXS;

  constructor(public payload: { ids: string[] }) {}
}

export class DeleteXxxxsByPredicate implements Action {
  readonly type = XxxxActionTypes.DELETE_XXXXS_BY_PREDICATE;

  constructor(public payload: { predicate: Predicate<Xxxx> }) {}
}

export class ClearXxxxs implements Action {
  readonly type = XxxxActionTypes.CLEAR_XXXXS;
}

export type XxxxActionsUnion =
  | LoadXxxxs
  | AddXxxx
  | UpsertXxxx
  | AddXxxxs
  | UpsertXxxxs
  | UpdateXxxx
  | UpdateXxxxs
  | MapXxxxs
  | DeleteXxxx
  | DeleteXxxxs
  | DeleteXxxxsByPredicate
  | ClearXxxxs;
    '''

    content = templateString.replace('Xxxx', model_class).replace('XXXX', model_name_upper).replace('xxxx', model_name)

    file_name = dir_name + "/" + "{}.actions.ts".format(model_name)

    file = open(file_name, "w")
    file.writelines(content)
    file.close()


def _create_actions(app):
    for model_class in _model_classes:
        _create_action(app, model_class)


def _create_reducer(app, model_class):
    model_name = model_class.lower()
    model_name_upper = model_class.upper()
    entity_path = app + '/' + model_name
    dir_name = _create_directory(entity_path, 'typescripts')

    templateString = r'''import { EntityState, EntityAdapter, createEntityAdapter } from '@ngrx/entity';
import { Xxxx } from './xxxx.model';
import { XxxxActionsUnion, XxxxActionTypes } from './xxxx.actions';

export interface State extends EntityState<Xxxx> {
  // additional entities state properties
  selectedXxxxId: number | null;
}

export const adapter: EntityAdapter<Xxxx> = createEntityAdapter<Xxxx>();

export const initialState: State = adapter.getInitialState({
  // additional entity state properties
  selectedXxxxId: null,
});

export function reducer(state = initialState, action: XxxxActionsUnion): State {
  switch (action.type) {
    case XxxxActionTypes.ADD_XXXX: {
      return adapter.addOne(action.payload.xxxx, state);
    }

    case XxxxActionTypes.UPSERT_XXXX: {
      return adapter.upsertOne(action.payload.xxxx, state);
    }

    case XxxxActionTypes.ADD_XXXXS: {
      return adapter.addMany(action.payload.xxxxs, state);
    }

    case XxxxActionTypes.UPSERT_XXXXS: {
      return adapter.upsertMany(action.payload.xxxxs, state);
    }

    case XxxxActionTypes.UPDATE_XXXX: {
      return adapter.updateOne(action.payload.xxxx, state);
    }

    case XxxxActionTypes.UPDATE_XXXXS: {
      return adapter.updateMany(action.payload.xxxxs, state);
    }

    case XxxxActionTypes.MAP_XXXXS: {
      return adapter.map(action.payload.entityMap, state);
    }

    case XxxxActionTypes.DELETE_XXXX: {
      return adapter.removeOne(action.payload.id, state);
    }

    case XxxxActionTypes.DELETE_XXXXS: {
      return adapter.removeMany(action.payload.ids, state);
    }

    case XxxxActionTypes.DELETE_XXXXS_BY_PREDICATE: {
      return adapter.removeMany(action.payload.predicate, state);
    }

    case XxxxActionTypes.LOAD_XXXXS: {
      return adapter.addAll(action.payload.xxxxs, state);
    }

    case XxxxActionTypes.CLEAR_XXXXS: {
      return adapter.removeAll({ ...state, selectedXxxxId: null });
    }

    default: {
      return state;
    }
  }
}

export const getSelectedXxxxId = (state: State) => state.selectedXxxxId;

// get the selectors
const {
  selectIds,
  selectEntities,
  selectAll,
  selectTotal,
} = adapter.getSelectors();

// select the array of xxxx ids
export const selectXxxxIds = selectIds;

// select the dictionary of xxxx entities
export const selectXxxxEntities = selectEntities;

// select the array of xxxxs
export const selectAllXxxxs = selectAll;

// select the total xxxx count
export const selectXxxxTotal = selectTotal;
    '''

    content = templateString.replace('Xxxx', model_class).replace('XXXX', model_name_upper).replace('xxxx', model_name)

    file_name = dir_name + "/" + "{}.reducer.ts".format(model_name)

    file = open(file_name, "w")
    file.writelines(content)
    file.close()


def _create_reducers(app):
    for model_class in _model_classes:
        _create_reducer(app, model_class)


def _create_selector(app, model_class):
    model_name = model_class.lower()
    model_name_upper = model_class.upper()
    entity_path = app + '/' + model_name
    dir_name = _create_directory(entity_path, 'typescripts')

    templateString = r'''import {
  createSelector,
  createFeatureSelector,
  ActionReducerMap,
} from '@ngrx/store';
import * as fromXxxx from './xxxx.reducer';

export interface State {
  xxxxs: fromXxxx.State;
}

export const reducers: ActionReducerMap<State> = {
  xxxxs: fromXxxx.reducer,
};

export const selectXxxxState = createFeatureSelector<fromXxxx.State>('xxxxs');

export const selectXxxxIds = createSelector(
  selectXxxxState,
  fromXxxx.selectXxxxIds
);
export const selectXxxxEntities = createSelector(
  selectXxxxState,
  fromXxxx.selectXxxxEntities
);
export const selectAllXxxxs = createSelector(
  selectXxxxState,
  fromXxxx.selectAllXxxxs
);
export const selectXxxxTotal = createSelector(
  selectXxxxState,
  fromXxxx.selectXxxxTotal
);
export const selectCurrentXxxxId = createSelector(
  selectXxxxState,
  fromXxxx.getSelectedXxxxId
);

export const selectCurrentXxxx = createSelector(
  selectXxxxEntities,
  selectCurrentXxxxId,
  (xxxxEntities, xxxxId) => xxxxEntities[xxxxId]
);
    '''

    content = templateString.replace('Xxxx', model_class).replace('XXXX', model_name_upper).replace('xxxx', model_name)

    file_name = dir_name + "/" + "{}.selectors.ts".format(model_name)

    file = open(file_name, "w")
    file.writelines(content)
    file.close()


def _create_selectors(app):
    for model_class in _model_classes:
        _create_selector(app, model_class)


def generate_api_code(app):
    _init_data(app)
    _create_package_init(app)
    _create_routes(app)
    _create_serializers(app)
    _create_urls(app)
    _create_views(app)


def generate_typescripts_code(app):
    _init_data(app)
    _create_directory(app, 'typescripts')  # create root directory
    _create_models(app)
    _create_actions(app)
    _create_reducers(app)
    _create_selectors(app)
